"""
Stocks Router — endpoints for stock list, detail, and zone analysis.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.services.data_service import DataService
from app.services.indicator_service import IndicatorService, INDICATOR_REGISTRY

router = APIRouter()
data_service = DataService()


# ============================================================
# HELPER: Build full zone analysis for a stock
# ============================================================

def build_zone_analysis(symbol: str):
    """Compute all indicators and zones for a stock."""
    stock = data_service.get_stock(symbol)
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock '{symbol}' not found")

    df = data_service.get_ohlcv(symbol)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No OHLCV data for '{symbol}'")

    # Compute all indicators
    zone_results = IndicatorService.compute_all(df, stock["price"], stock)

    # Build response
    zones = []
    total_score = 0
    buy_count = 0
    sell_count = 0
    neutral_count = 0

    for ind_id, result in zone_results.items():
        meta = INDICATOR_REGISTRY.get(ind_id, {})
        zone_item = {
            "id": ind_id,
            "name": meta.get("name", ind_id),
            "full_name": meta.get("full_name", ind_id),
            "category": meta.get("category", "Other"),
            "zone": {
                "zone": result["zone"],
                "signal": result["signal"],
                "color": result["color"],
                "score": result["score"],
                "detail": result["detail"],
                "value": result["value"],
                "education": meta.get("education", ""),
            },
        }
        zones.append(zone_item)
        total_score += result["score"]
        if result["score"] > 0:
            buy_count += 1
        elif result["score"] < 0:
            sell_count += 1
        else:
            neutral_count += 1

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

    return {
        "stock": stock,
        "zones": zones,
        "consensus_score": total_score,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "neutral_count": neutral_count,
        "consensus_label": consensus,
    }


# ============================================================
# ENDPOINTS
# ============================================================

@router.get("/")
async def list_stocks(
    sector: Optional[str] = Query(None, description="Filter by sector"),
    search: Optional[str] = Query(None, description="Search by symbol or name"),
    sort_by: str = Query("symbol", description="Sort by: symbol, change_pct, consensus_score"),
    sort_order: str = Query("asc", description="asc or desc"),
    limit: int = Query(50, le=100),
):
    """
    List all stocks with compact zone summary.
    
    Returns a lightweight list suitable for the main table view.
    Each stock includes price, change, and a consensus score.
    """
    if data_service.needs_refresh:
        await data_service.refresh_all()

    stocks = data_service.get_all_stocks()

    # Apply filters
    if sector:
        stocks = [s for s in stocks if s["sector"].lower() == sector.lower()]
    if search:
        q = search.lower()
        stocks = [s for s in stocks if q in s["symbol"].lower() or q in s["name"].lower()]

    # Compute zones for each stock (compact)
    results = []
    for stock in stocks:
        try:
            analysis = build_zone_analysis(stock["symbol"])
            results.append({
                "symbol": stock["symbol"],
                "name": stock["name"],
                "sector": stock["sector"],
                "price": stock["price"],
                "change_pct": stock["change_pct"],
                "consensus_score": analysis["consensus_score"],
                "buy_count": analysis["buy_count"],
                "sell_count": analysis["sell_count"],
                "neutral_count": analysis["neutral_count"],
                "consensus_label": analysis["consensus_label"],
                "zone_summary": [
                    {
                        "id": z["id"],
                        "name": z["name"],
                        "signal": z["zone"]["signal"],
                        "color": z["zone"]["color"],
                        "score": z["zone"]["score"],
                    }
                    for z in analysis["zones"]
                ],
            })
        except Exception:
            continue

    # Sort
    reverse = sort_order.lower() == "desc"
    if sort_by == "consensus_score":
        results.sort(key=lambda x: x["consensus_score"], reverse=reverse)
    elif sort_by == "change_pct":
        results.sort(key=lambda x: x["change_pct"], reverse=reverse)
    else:
        results.sort(key=lambda x: x["symbol"], reverse=reverse)

    return {
        "total": len(results),
        "stocks": results[:limit],
        "sectors": data_service.get_sectors(),
    }

@router.get("/lookup/{symbol}")
async def lookup_stock(symbol: str):
    symbol = symbol.upper().strip()
    stock = data_service.get_stock(symbol)
    if not stock:
        stock = await data_service.fetch_on_demand(symbol)
        if not stock:
            raise HTTPException(
                status_code=404,
                detail=f"Stock '{symbol}' not found on NSE."
            )
    return build_zone_analysis(symbol)

@router.delete("/lookup/{symbol}")
async def remove_custom_stock(symbol: str):
    removed = data_service.remove_custom_stock(symbol.upper())
    if removed:
        return {"message": f"'{symbol}' removed"}
    raise HTTPException(status_code=404, detail="Not a custom stock")
@router.get("/{symbol}")
async def get_stock(symbol: str):
    """
    Get stock price details.
    """
    stock = data_service.get_stock(symbol.upper())
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock '{symbol}' not found")
    return stock


@router.get("/{symbol}/zones")
async def get_stock_zones(symbol: str):
    """
    Get full zone analysis for a stock.
    
    Returns all technical indicator zones with scores,
    signals, educational tooltips, and a consensus rating.
    This powers the stock detail slide-out panel.
    """
    return build_zone_analysis(symbol.upper())
