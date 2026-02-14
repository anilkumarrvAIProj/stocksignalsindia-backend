"""
Scanner Router — screen stocks based on indicator conditions.

Examples:
- Find all stocks where RSI < 30 (oversold)
- Find stocks in Bollinger lower band AND MACD bullish
- Find stocks with consensus score > 5 (strong buy)
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.services.data_service import DataService
from app.services.indicator_service import IndicatorService, INDICATOR_REGISTRY

router = APIRouter()
data_service = DataService()


@router.get("/scan")
async def scan_stocks(
    # Indicator-specific filters
    indicator: Optional[str] = Query(None, description="Filter by specific indicator id (e.g., 'rsi', 'bollinger')"),
    signal: Optional[str] = Query(None, description="Filter by signal (BUY, SELL, HOLD, etc.)"),
    zone: Optional[str] = Query(None, description="Filter by zone name (e.g., 'Oversold', 'Golden Cross')"),
    
    # Score filters
    min_score: Optional[int] = Query(None, description="Min consensus score (-20 to 20)"),
    max_score: Optional[int] = Query(None, description="Max consensus score (-20 to 20)"),
    
    # Stock filters
    sector: Optional[str] = Query(None, description="Filter by sector"),
    min_change: Optional[float] = Query(None, description="Min % change"),
    max_change: Optional[float] = Query(None, description="Max % change"),
    
    # Sorting
    sort_by: str = Query("consensus_score", description="Sort field"),
    sort_order: str = Query("desc", description="asc or desc"),
    limit: int = Query(50, le=100),
):
    """
    Scan/screen stocks based on multiple criteria.
    
    This is the power-user feature — combine indicator zones,
    consensus scores, sectors, and price changes to find
    stocks matching your exact criteria.
    
    Examples:
    - /scan?indicator=rsi&signal=BUY → All stocks where RSI says BUY
    - /scan?min_score=5 → All stocks with strong buy consensus
    - /scan?indicator=bollinger&zone=Lower Band&sector=Banking
    - /scan?min_change=-5&max_change=0&min_score=3 → Dipped stocks with bullish indicators
    """
    stocks = data_service.get_all_stocks()
    
    # Apply sector filter early
    if sector:
        stocks = [s for s in stocks if s["sector"].lower() == sector.lower()]

    if min_change is not None:
        stocks = [s for s in stocks if s["change_pct"] >= min_change]
    if max_change is not None:
        stocks = [s for s in stocks if s["change_pct"] <= max_change]

    results = []

    for stock in stocks:
        try:
            df = data_service.get_ohlcv(stock["symbol"])
            if df is None or df.empty:
                continue

            all_zones = IndicatorService.compute_all(df, stock["price"], stock)

            # Calculate consensus
            total_score = sum(z["score"] for z in all_zones.values())
            buy_count = sum(1 for z in all_zones.values() if z["score"] > 0)
            sell_count = sum(1 for z in all_zones.values() if z["score"] < 0)
            neutral_count = sum(1 for z in all_zones.values() if z["score"] == 0)

            # Apply score filters
            if min_score is not None and total_score < min_score:
                continue
            if max_score is not None and total_score > max_score:
                continue

            # Apply indicator-specific filters
            if indicator:
                if indicator not in all_zones:
                    continue
                ind_zone = all_zones[indicator]
                if signal and ind_zone["signal"].upper() != signal.upper():
                    continue
                if zone and ind_zone["zone"].lower() != zone.lower():
                    continue

            # Consensus label
            if total_score > 6:
                consensus = "Strong Buy"
            elif total_score > 2:
                consensus = "Buy"
            elif total_score >= -2:
                consensus = "Neutral"
            elif total_score >= -6:
                consensus = "Sell"
            else:
                consensus = "Strong Sell"

            results.append({
                "symbol": stock["symbol"],
                "name": stock["name"],
                "sector": stock["sector"],
                "price": stock["price"],
                "change_pct": stock["change_pct"],
                "consensus_score": total_score,
                "buy_count": buy_count,
                "sell_count": sell_count,
                "neutral_count": neutral_count,
                "consensus_label": consensus,
                "zone_summary": [
                    {
                        "id": ind_id,
                        "name": INDICATOR_REGISTRY.get(ind_id, {}).get("name", ind_id),
                        "signal": z["signal"],
                        "zone": z["zone"],
                        "color": z["color"],
                        "score": z["score"],
                    }
                    for ind_id, z in all_zones.items()
                ],
                # Include the specific indicator detail if filtered
                "filtered_indicator": all_zones.get(indicator) if indicator else None,
            })

        except Exception as e:
            continue

    # Sort
    reverse = sort_order.lower() == "desc"
    if sort_by == "consensus_score":
        results.sort(key=lambda x: x["consensus_score"], reverse=reverse)
    elif sort_by == "change_pct":
        results.sort(key=lambda x: x["change_pct"], reverse=reverse)
    elif sort_by == "buy_count":
        results.sort(key=lambda x: x["buy_count"], reverse=reverse)
    else:
        results.sort(key=lambda x: x["symbol"], reverse=reverse)

    filters_applied = {
        k: v for k, v in {
            "indicator": indicator, "signal": signal, "zone": zone,
            "min_score": min_score, "max_score": max_score,
            "sector": sector, "min_change": min_change, "max_change": max_change,
        }.items() if v is not None
    }

    return {
        "total": len(results),
        "results": results[:limit],
        "filters_applied": filters_applied,
    }


@router.get("/presets")
async def get_scan_presets():
    """
    Get pre-built scanner presets for common strategies.
    
    Returns a list of scan configurations that users can apply with one click.
    """
    return {
        "presets": [
            {
                "name": "Oversold Bounce Candidates",
                "description": "Stocks with RSI below 30 — potential reversal plays",
                "params": {"indicator": "rsi", "signal": "BUY", "sort_by": "consensus_score"},
            },
            {
                "name": "Bollinger Squeeze — Ready to Break",
                "description": "Stocks near Bollinger lower band with bullish MACD",
                "params": {"indicator": "bollinger", "zone": "Lower Band"},
            },
            {
                "name": "Strong Buy Consensus",
                "description": "Stocks where 6+ indicators agree on BUY",
                "params": {"min_score": 6, "sort_by": "consensus_score", "sort_order": "desc"},
            },
            {
                "name": "Golden Cross Stocks",
                "description": "50-day SMA crossed above 200-day SMA",
                "params": {"indicator": "ma_cross", "zone": "Golden Cross"},
            },
            {
                "name": "Supertrend Bullish",
                "description": "Stocks where Supertrend confirms uptrend",
                "params": {"indicator": "supertrend", "signal": "BUY"},
            },
            {
                "name": "Sell Candidates",
                "description": "Stocks with strong sell signals across multiple indicators",
                "params": {"max_score": -4, "sort_by": "consensus_score", "sort_order": "asc"},
            },
            {
                "name": "Dip Buying Opportunities",
                "description": "Stocks down 2%+ today but with bullish technical signals",
                "params": {"max_change": -2, "min_score": 2, "sort_by": "consensus_score"},
            },
            {
                "name": "MACD Bullish Crossover",
                "description": "Stocks where MACD just crossed above signal line",
                "params": {"indicator": "macd", "signal": "BUY"},
            },
        ]
    }
