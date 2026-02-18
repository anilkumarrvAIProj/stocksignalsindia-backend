"""
Indicators Router — get individual indicator data for a stock.
"""

from fastapi import APIRouter, HTTPException

from app.services.data_service import DataService
from app.services.indicator_service import IndicatorService, INDICATOR_REGISTRY

router = APIRouter()
data_service = DataService()


@router.get("/")
async def list_indicators():
    """
    List all available technical indicators with metadata.
    """
    return {
        "indicators": [
            {
                "id": ind_id,
                "name": meta["name"],
                "full_name": meta["full_name"],
                "category": meta["category"],
                "education": meta["education"],
            }
            for ind_id, meta in INDICATOR_REGISTRY.items()
        ]
    }


@router.get("/{symbol}/{indicator_id}")
async def get_indicator(symbol: str, indicator_id: str):
    """
    Get a specific indicator's zone analysis for a stock.
    
    Returns the full zone result including raw values,
    educational explanation, and signal.
    """
    symbol = symbol.upper()
    stock = data_service.get_stock(symbol)
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock '{symbol}' not found")

    df = data_service.get_ohlcv(symbol)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No data for '{symbol}'")

    if indicator_id not in INDICATOR_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown indicator '{indicator_id}'. Available: {list(INDICATOR_REGISTRY.keys())}",
        )

    # Compute all and extract the requested one
    all_zones = IndicatorService.compute_all(df, stock["price"], stock)
    
    if indicator_id not in all_zones:
        raise HTTPException(status_code=500, detail=f"Failed to compute '{indicator_id}'")

    result = all_zones[indicator_id]
    meta = INDICATOR_REGISTRY[indicator_id]

    return {
        "stock": {"symbol": symbol, "price": stock["price"], "change_pct": stock["change_pct"]},
        "indicator": {
            "id": indicator_id,
            "name": meta["name"],
            "full_name": meta["full_name"],
            "category": meta["category"],
            "education": meta["education"],
        },
        "zone": result,
    }
