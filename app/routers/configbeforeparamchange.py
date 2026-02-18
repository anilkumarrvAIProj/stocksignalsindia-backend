"""
Config Router — API endpoints for indicator configuration.

Users can:
1. View all indicator parameters and their defaults
2. Apply presets (Aggressive, Conservative, Intraday)
3. Customize individual indicator parameters
4. Get zones recalculated with custom config
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.services.indicator_config import (
    DEFAULT_INDICATOR_CONFIG,
    INDICATOR_PRESETS,
    get_config_with_overrides,
    apply_preset,
)
from app.services.data_service import DataService
from app.services.indicator_service import IndicatorService, INDICATOR_REGISTRY

router = APIRouter()
data_service = DataService()

# In-memory user config (in production, store per-user in DB/Redis)
_user_config: dict = {}


# ============================================================
# REQUEST MODELS
# ============================================================

class ConfigUpdate(BaseModel):
    """Update config for one or more indicators."""
    config: dict  # {indicator_id: {param_name: value}}


class PresetApply(BaseModel):
    """Apply a preset configuration."""
    preset: str  # "default", "aggressive", "conservative", "intraday"


# ============================================================
# ENDPOINTS
# ============================================================

@router.get("/")
async def get_all_config():
    """
    Get all indicator configurations with current values,
    defaults, min/max bounds, and descriptions.
    
    This powers the Settings UI.
    """
    current = get_config_with_overrides(_user_config)

    result = {}
    for ind_id, ind_meta in DEFAULT_INDICATOR_CONFIG.items():
        params = {}
        for param_name, param_info in ind_meta["params"].items():
            params[param_name] = {
                **param_info,
                "current_value": current[ind_id][param_name],
                "is_default": current[ind_id][param_name] == param_info["default"],
            }
        result[ind_id] = {
            "name": ind_meta["name"],
            "full_name": ind_meta["full_name"],
            "category": ind_meta["category"],
            "params": params,
        }

    return {
        "indicators": result,
        "active_config": current,
    }


@router.get("/presets")
async def get_presets():
    """
    Get all available preset configurations.
    """
    return {
        "presets": {
            name: {
                "name": preset["name"],
                "description": preset["description"],
                "config": preset["config"],
            }
            for name, preset in INDICATOR_PRESETS.items()
        }
    }


@router.post("/presets/{preset_name}")
async def apply_preset_endpoint(preset_name: str):
    """
    Apply a preset configuration.
    
    Available presets: default, aggressive, conservative, intraday
    """
    global _user_config

    if preset_name not in INDICATOR_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown preset '{preset_name}'. Available: {list(INDICATOR_PRESETS.keys())}",
        )

    if preset_name == "default":
        _user_config = {}
    else:
        _user_config = INDICATOR_PRESETS[preset_name]["config"]

    return {
        "message": f"Preset '{preset_name}' applied successfully",
        "preset": INDICATOR_PRESETS[preset_name]["name"],
        "active_config": get_config_with_overrides(_user_config),
    }


@router.post("/update")
async def update_config(body: ConfigUpdate):
    """
    Update indicator configuration with custom values.
    
    Body example:
    {
        "config": {
            "rsi": {"period": 7, "overbought": 75, "oversold": 25},
            "bollinger": {"period": 15, "std_dev": 2.5}
        }
    }
    
    Only send the indicators/params you want to change.
    Others will keep their current values.
    """
    global _user_config

    # Merge new config over existing user config
    for ind_id, params in body.config.items():
        if ind_id not in DEFAULT_INDICATOR_CONFIG:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown indicator '{ind_id}'. Available: {list(DEFAULT_INDICATOR_CONFIG.keys())}",
            )

        if ind_id not in _user_config:
            _user_config[ind_id] = {}

        for param_name, value in params.items():
            if param_name not in DEFAULT_INDICATOR_CONFIG[ind_id]["params"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown param '{param_name}' for '{ind_id}'. Available: {list(DEFAULT_INDICATOR_CONFIG[ind_id]['params'].keys())}",
                )
            _user_config[ind_id][param_name] = value

    return {
        "message": "Configuration updated",
        "active_config": get_config_with_overrides(_user_config),
    }


@router.post("/reset")
async def reset_config():
    """Reset all configurations to defaults."""
    global _user_config
    _user_config = {}
    return {
        "message": "Configuration reset to defaults",
        "active_config": get_config_with_overrides({}),
    }


@router.post("/reset/{indicator_id}")
async def reset_indicator_config(indicator_id: str):
    """Reset a single indicator to its default configuration."""
    global _user_config

    if indicator_id not in DEFAULT_INDICATOR_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unknown indicator '{indicator_id}'")

    if indicator_id in _user_config:
        del _user_config[indicator_id]

    return {
        "message": f"'{indicator_id}' reset to defaults",
        "active_config": get_config_with_overrides(_user_config),
    }


@router.get("/preview/{symbol}")
async def preview_with_config(symbol: str):
    """
    Preview zone analysis for a stock with the CURRENT config.
    Use this to see how config changes affect signals before committing.
    """
    symbol = symbol.upper()
    stock = data_service.get_stock(symbol)
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock '{symbol}' not found")

    df = data_service.get_ohlcv(symbol)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No data for '{symbol}'")

    # Compute with current user config
    zone_results = IndicatorService.compute_all(df, stock["price"], stock, config=_user_config)

    zones = []
    total_score = 0

    for ind_id, result in zone_results.items():
        meta = INDICATOR_REGISTRY.get(ind_id, {})
        zones.append({
            "id": ind_id,
            "name": meta.get("name", ind_id),
            "zone": result["zone"],
            "signal": result["signal"],
            "color": result["color"],
            "score": result["score"],
            "detail": result["detail"],
            "config_used": result.get("config_used", {}),
        })
        total_score += result["score"]

    return {
        "symbol": symbol,
        "price": stock["price"],
        "consensus_score": total_score,
        "zones": zones,
        "config_applied": get_config_with_overrides(_user_config),
    }


def get_current_user_config():
    """Utility function for other routers to access current config."""
    return _user_config
