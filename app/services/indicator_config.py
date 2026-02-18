"""
Indicator Configuration — User-customizable parameters for each indicator.

Each indicator has default values that match industry standards,
but users can override any parameter via the API or the settings UI.
"""

# ============================================================
# DEFAULT CONFIGURATIONS
# ============================================================

DEFAULT_INDICATOR_CONFIG = {
    "rsi": {
        "name": "RSI",
        "full_name": "Relative Strength Index",
        "category": "Momentum",
        "params": {
            "period": {
                "label": "Period",
                "description": "Number of candles to calculate RSI",
                "type": "int",
                "default": 14,
                "min": 2,
                "max": 100,
                "step": 1,
            },
            "overbought": {
                "label": "Overbought Level",
                "description": "RSI above this = overbought zone",
                "type": "float",
                "default": 70,
                "min": 50,
                "max": 95,
                "step": 1,
            },
            "oversold": {
                "label": "Oversold Level",
                "description": "RSI below this = oversold zone",
                "type": "float",
                "default": 30,
                "min": 5,
                "max": 50,
                "step": 1,
            },
        },
    },
    "bollinger": {
        "name": "Bollinger Bands",
        "full_name": "Bollinger Bands",
        "category": "Volatility",
        "params": {
            "period": {
                "label": "Period",
                "description": "Moving average period for middle band",
                "type": "int",
                "default": 20,
                "min": 5,
                "max": 100,
                "step": 1,
            },
            "std_dev": {
                "label": "Std Deviation",
                "description": "Number of standard deviations for upper/lower bands",
                "type": "float",
                "default": 2.0,
                "min": 0.5,
                "max": 4.0,
                "step": 0.1,
            },
        },
    },
    "macd": {
        "name": "MACD",
        "full_name": "Moving Average Convergence Divergence",
        "category": "Trend",
        "params": {
            "fast_period": {
                "label": "Fast EMA",
                "description": "Period for the fast EMA line",
                "type": "int",
                "default": 12,
                "min": 2,
                "max": 50,
                "step": 1,
            },
            "slow_period": {
                "label": "Slow EMA",
                "description": "Period for the slow EMA line",
                "type": "int",
                "default": 26,
                "min": 10,
                "max": 100,
                "step": 1,
            },
            "signal_period": {
                "label": "Signal Line",
                "description": "Period for the signal line EMA",
                "type": "int",
                "default": 9,
                "min": 2,
                "max": 50,
                "step": 1,
            },
        },
    },
    "pivot": {
        "name": "Pivot Points",
        "full_name": "Pivot Points",
        "category": "Support/Resistance",
        "params": {
            "method": {
                "label": "Calculation Method",
                "description": "Formula used to calculate pivot levels",
                "type": "select",
                "default": "standard",
                "options": [
                    {"value": "standard", "label": "Standard"},
                    {"value": "fibonacci", "label": "Fibonacci"},
                    {"value": "camarilla", "label": "Camarilla"},
                    {"value": "woodie", "label": "Woodie"},
                ],
            },
        },
    },
    "supertrend": {
        "name": "Supertrend",
        "full_name": "Supertrend",
        "category": "Trend",
        "params": {
            "period": {
                "label": "ATR Period",
                "description": "Period for ATR calculation",
                "type": "int",
                "default": 10,
                "min": 5,
                "max": 50,
                "step": 1,
            },
            "multiplier": {
                "label": "Multiplier",
                "description": "ATR multiplier for band width",
                "type": "float",
                "default": 3.0,
                "min": 1.0,
                "max": 6.0,
                "step": 0.1,
            },
        },
    },
    "stochastic": {
        "name": "Stochastic",
        "full_name": "Stochastic Oscillator",
        "category": "Momentum",
        "params": {
            "k_period": {
                "label": "%K Period",
                "description": "Lookback period for %K line",
                "type": "int",
                "default": 14,
                "min": 5,
                "max": 50,
                "step": 1,
            },
            "d_period": {
                "label": "%D Period",
                "description": "Smoothing period for %D signal line",
                "type": "int",
                "default": 3,
                "min": 2,
                "max": 20,
                "step": 1,
            },
            "overbought": {
                "label": "Overbought Level",
                "description": "%K above this = overbought",
                "type": "float",
                "default": 80,
                "min": 60,
                "max": 95,
                "step": 1,
            },
            "oversold": {
                "label": "Oversold Level",
                "description": "%K below this = oversold",
                "type": "float",
                "default": 20,
                "min": 5,
                "max": 40,
                "step": 1,
            },
        },
    },
    "adx": {
        "name": "ADX",
        "full_name": "Average Directional Index",
        "category": "Trend",
        "params": {
            "period": {
                "label": "Period",
                "description": "Lookback period for ADX",
                "type": "int",
                "default": 14,
                "min": 5,
                "max": 50,
                "step": 1,
            },
            "strong_trend": {
                "label": "Strong Trend Level",
                "description": "ADX above this = strong trend",
                "type": "float",
                "default": 25,
                "min": 15,
                "max": 50,
                "step": 1,
            },
        },
    },
    "cci": {
        "name": "CCI",
        "full_name": "Commodity Channel Index",
        "category": "Momentum",
        "params": {
            "period": {
                "label": "Period",
                "description": "Lookback period for CCI",
                "type": "int",
                "default": 20,
                "min": 5,
                "max": 100,
                "step": 1,
            },
            "overbought": {
                "label": "Overbought Level",
                "description": "CCI above this = overbought",
                "type": "float",
                "default": 100,
                "min": 50,
                "max": 300,
                "step": 10,
            },
            "oversold": {
                "label": "Oversold Level",
                "description": "CCI below this = oversold",
                "type": "float",
                "default": -100,
                "min": -300,
                "max": -50,
                "step": 10,
            },
        },
    },
    "vwap": {
        "name": "VWAP",
        "full_name": "Volume Weighted Average Price",
        "category": "Volume",
        "params": {
            "threshold_pct": {
                "label": "Threshold %",
                "description": "% distance from VWAP to consider significant",
                "type": "float",
                "default": 2.0,
                "min": 0.5,
                "max": 5.0,
                "step": 0.5,
            },
        },
    },
    "ma_cross": {
        "name": "MA Cross",
        "full_name": "Moving Average Crossover",
        "category": "Trend",
        "params": {
            "fast_period": {
                "label": "Fast SMA",
                "description": "Period for the fast moving average",
                "type": "int",
                "default": 50,
                "min": 5,
                "max": 100,
                "step": 5,
            },
            "slow_period": {
                "label": "Slow SMA",
                "description": "Period for the slow moving average",
                "type": "int",
                "default": 200,
                "min": 50,
                "max": 500,
                "step": 10,
            },
        },
    },
    "williams_r": {
        "name": "Williams %R",
        "full_name": "Williams Percent Range",
        "category": "Momentum",
        "params": {
            "period": {
                "label": "Period",
                "description": "Lookback period",
                "type": "int",
                "default": 14,
                "min": 5,
                "max": 50,
                "step": 1,
            },
            "overbought": {
                "label": "Overbought Level",
                "description": "Above this = overbought (e.g., -20)",
                "type": "float",
                "default": -20,
                "min": -40,
                "max": -5,
                "step": 1,
            },
            "oversold": {
                "label": "Oversold Level",
                "description": "Below this = oversold (e.g., -80)",
                "type": "float",
                "default": -80,
                "min": -95,
                "max": -60,
                "step": 1,
            },
        },
    },
    "atr": {
        "name": "ATR",
        "full_name": "Average True Range",
        "category": "Volatility",
        "params": {
            "period": {
                "label": "Period",
                "description": "Lookback period for ATR",
                "type": "int",
                "default": 14,
                "min": 5,
                "max": 50,
                "step": 1,
            },
            "high_volatility_pct": {
                "label": "High Volatility %",
                "description": "ATR as % of price above this = high volatility",
                "type": "float",
                "default": 4.0,
                "min": 1.0,
                "max": 10.0,
                "step": 0.5,
            },
        },
    },
    "ema_cross": {
        "name": "EMA Cross",
        "full_name": "EMA Crossover",
        "category": "Trend",
        "params": {
            "fast_period": {
                "label": "Fast EMA",
                "description": "Period for fast EMA",
                "type": "int",
                "default": 12,
                "min": 3,
                "max": 50,
                "step": 1,
            },
            "slow_period": {
                "label": "Slow EMA",
                "description": "Period for slow EMA",
                "type": "int",
                "default": 26,
                "min": 10,
                "max": 100,
                "step": 1,
            },
        },
    },
    "obv": {
        "name": "OBV",
        "full_name": "On-Balance Volume",
        "category": "Volume",
        "params": {
            "lookback_days": {
                "label": "Lookback Days",
                "description": "Days to compare OBV trend",
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 30,
                "step": 1,
            },
        },
    },
}


# ============================================================
# POPULAR PRESETS — One-click configurations
# ============================================================

INDICATOR_PRESETS = {
    "default": {
        "name": "Standard (Default)",
        "description": "Industry-standard settings used by most platforms",
        "config": {},  # Empty = use all defaults
    },
    "aggressive": {
        "name": "Aggressive / Scalping",
        "description": "Shorter periods for faster signals — more trades, more noise",
        "config": {
            "rsi": {"period": 7, "overbought": 75, "oversold": 25},
            "bollinger": {"period": 10, "std_dev": 1.5},
            "macd": {"fast_period": 8, "slow_period": 17, "signal_period": 9},
            "supertrend": {"period": 7, "multiplier": 2.0},
            "stochastic": {"k_period": 9, "d_period": 3, "overbought": 80, "oversold": 20},
            "adx": {"period": 10, "strong_trend": 20},
            "cci": {"period": 14, "overbought": 100, "oversold": -100},
            "ma_cross": {"fast_period": 20, "slow_period": 50},
            "ema_cross": {"fast_period": 8, "slow_period": 21},
            "williams_r": {"period": 7},
            "atr": {"period": 10},
            "obv": {"lookback_days": 3},
        },
    },
    "conservative": {
        "name": "Conservative / Swing",
        "description": "Longer periods for fewer but more reliable signals",
        "config": {
            "rsi": {"period": 21, "overbought": 65, "oversold": 35},
            "bollinger": {"period": 30, "std_dev": 2.5},
            "macd": {"fast_period": 19, "slow_period": 39, "signal_period": 9},
            "supertrend": {"period": 14, "multiplier": 4.0},
            "stochastic": {"k_period": 21, "d_period": 5, "overbought": 85, "oversold": 15},
            "adx": {"period": 20, "strong_trend": 30},
            "cci": {"period": 30, "overbought": 150, "oversold": -150},
            "ma_cross": {"fast_period": 50, "slow_period": 200},
            "ema_cross": {"fast_period": 20, "slow_period": 50},
            "williams_r": {"period": 21},
            "atr": {"period": 20},
            "obv": {"lookback_days": 10},
        },
    },
    "intraday": {
        "name": "Intraday / Day Trading",
        "description": "Optimized for intraday moves — very fast signals",
        "config": {
            "rsi": {"period": 5, "overbought": 80, "oversold": 20},
            "bollinger": {"period": 10, "std_dev": 2.0},
            "macd": {"fast_period": 5, "slow_period": 13, "signal_period": 8},
            "supertrend": {"period": 5, "multiplier": 2.0},
            "stochastic": {"k_period": 5, "d_period": 3, "overbought": 80, "oversold": 20},
            "adx": {"period": 7, "strong_trend": 20},
            "cci": {"period": 10, "overbought": 100, "oversold": -100},
            "ma_cross": {"fast_period": 9, "slow_period": 21},
            "ema_cross": {"fast_period": 5, "slow_period": 13},
            "williams_r": {"period": 5},
            "atr": {"period": 7},
            "obv": {"lookback_days": 1},
        },
    },
}


# ============================================================
# INDICATOR REGISTRY — metadata + education text
# ============================================================

INDICATOR_REGISTRY = {
    "rsi": {
        "name": "RSI", "full_name": "Relative Strength Index", "category": "Momentum",
        "education": "RSI measures the speed and magnitude of price changes on a scale of 0-100. Values above the overbought level suggest the stock may be overpriced. Values below the oversold level suggest it may be undervalued.",
    },
    "bollinger": {
        "name": "Bollinger Bands", "full_name": "Bollinger Bands", "category": "Volatility",
        "education": "Bollinger Bands consist of a middle band (SMA) and upper/lower bands at N standard deviations. When price touches the upper band, the stock may be overbought. A squeeze often precedes a big move.",
    },
    "macd": {
        "name": "MACD", "full_name": "Moving Avg Convergence Divergence", "category": "Trend",
        "education": "MACD tracks the relationship between a fast and slow EMA. The MACD line crossing above the signal line is bullish. The histogram shows signal strength.",
    },
    "pivot": {
        "name": "Pivot Points", "full_name": "Pivot Points", "category": "Support/Resistance",
        "education": "Pivot Points define support and resistance levels from the previous day's price action. Multiple methods available: Standard, Fibonacci, Camarilla, Woodie.",
    },
    "supertrend": {
        "name": "Supertrend", "full_name": "Supertrend", "category": "Trend",
        "education": "Supertrend is a trend-following indicator based on ATR. Price above the line = bullish, below = bearish. Adjust the period and multiplier to control sensitivity.",
    },
    "stochastic": {
        "name": "Stochastic", "full_name": "Stochastic Oscillator", "category": "Momentum",
        "education": "Stochastic compares the closing price to the price range. %K above overbought level is overbought, below oversold level is oversold.",
    },
    "adx": {
        "name": "ADX", "full_name": "Average Directional Index", "category": "Trend",
        "education": "ADX measures trend strength regardless of direction. Above the strong trend level indicates a strong trend. It doesn't tell direction — combine with DI lines.",
    },
    "cci": {
        "name": "CCI", "full_name": "Commodity Channel Index", "category": "Momentum",
        "education": "CCI measures how far price has deviated from its statistical mean. Extreme readings suggest overbought/oversold conditions.",
    },
    "vwap": {
        "name": "VWAP", "full_name": "Volume Weighted Average Price", "category": "Volume",
        "education": "VWAP is the average price weighted by volume. Institutional traders use it as a benchmark. Above VWAP is bullish, below is bearish.",
    },
    "ma_cross": {
        "name": "MA Cross", "full_name": "Moving Average Crossover", "category": "Trend",
        "education": "When the fast SMA crosses above the slow SMA, it forms a Golden Cross (bullish). When it crosses below, it forms a Death Cross (bearish).",
    },
    "williams_r": {
        "name": "Williams %R", "full_name": "Williams Percent Range", "category": "Momentum",
        "education": "Williams %R ranges from -100 to 0. Readings above the overbought level indicate overbought, below oversold level indicate oversold.",
    },
    "atr": {
        "name": "ATR", "full_name": "Average True Range", "category": "Volatility",
        "education": "ATR measures market volatility. Higher ATR means higher volatility. Useful for setting stop-losses and position sizing.",
    },
    "ema_cross": {
        "name": "EMA Cross", "full_name": "EMA Crossover", "category": "Trend",
        "education": "EMA crossover is faster than SMA crossover. EMA gives more weight to recent prices. Fast EMA crossing above slow EMA is bullish.",
    },
    "obv": {
        "name": "OBV", "full_name": "On-Balance Volume", "category": "Volume",
        "education": "OBV tracks cumulative volume flow. Rising OBV confirms uptrend, falling confirms downtrend. Divergence from price signals reversals.",
    },
}


def get_config_with_overrides(user_config: dict = None) -> dict:
    """
    Merge user overrides with defaults.
    
    Args:
        user_config: Dict of {indicator_id: {param_name: value}}
        
    Returns:
        Full config dict with user values merged over defaults
    """
    result = {}

    for ind_id, ind_meta in DEFAULT_INDICATOR_CONFIG.items():
        result[ind_id] = {}
        for param_name, param_meta in ind_meta["params"].items():
            # Start with default
            value = param_meta["default"]

            # Apply user override if present
            if user_config and ind_id in user_config and param_name in user_config[ind_id]:
                user_val = user_config[ind_id][param_name]
                # Validate bounds
                if param_meta["type"] in ("int", "float"):
                    value = max(param_meta["min"], min(param_meta["max"], user_val))
                elif param_meta["type"] == "select":
                    valid = [o["value"] for o in param_meta["options"]]
                    if user_val in valid:
                        value = user_val
                else:
                    value = user_val

            result[ind_id][param_name] = value

    return result


def apply_preset(preset_name: str) -> dict:
    """Apply a preset configuration."""
    preset = INDICATOR_PRESETS.get(preset_name)
    if not preset:
        return get_config_with_overrides()
    return get_config_with_overrides(preset["config"])
# ============================================================
# RUNTIME CONFIG STORAGE (In-Memory)
# ============================================================

# Holds current user overrides
_CURRENT_USER_CONFIG = {}


def get_current_config() -> dict:
    """
    Return the currently active user configuration.
    """
    return _CURRENT_USER_CONFIG


def update_current_config(new_config: dict):
    """
    Replace current user config with new values.
    Expected format:
    {
        "rsi": {"period": 50},
        "macd": {"fast_period": 8}
    }
    """
    global _CURRENT_USER_CONFIG
    _CURRENT_USER_CONFIG = new_config or {}
