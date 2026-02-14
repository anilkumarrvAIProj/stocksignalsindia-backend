"""
IndicatorService — Computes all technical indicators with CONFIGURABLE parameters.

Uses the `ta` library (https://github.com/bukosabino/ta).
Compatible with Python 3.9+.

All indicator parameters (periods, thresholds, etc.) can be customized
via the config dict passed to compute_all().
"""

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD, ADXIndicator, EMAIndicator, SMAIndicator
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
from typing import Optional
import logging

from app.services.indicator_config import (
    DEFAULT_INDICATOR_CONFIG,
    INDICATOR_REGISTRY,
    get_config_with_overrides,
)

logger = logging.getLogger(__name__)


# Re-export INDICATOR_REGISTRY so existing imports still work
# (it now lives in indicator_config.py)


# ============================================================
# SUPERTREND — Manual implementation (not in `ta` library)
# ============================================================

def compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    """
    Manual Supertrend calculation.
    Returns (supertrend_value, direction) for the latest bar.
    direction: 1 = bullish, -1 = bearish
    """
    hl2 = (df["High"] + df["Low"]) / 2
    atr_indicator = AverageTrueRange(df["High"], df["Low"], df["Close"], window=period)
    atr = atr_indicator.average_true_range()

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    supertrend.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = -1

    for i in range(1, len(df)):
        if lower_band.iloc[i] > lower_band.iloc[i - 1] or df["Close"].iloc[i - 1] < lower_band.iloc[i - 1]:
            pass
        else:
            lower_band.iloc[i] = lower_band.iloc[i - 1]

        if upper_band.iloc[i] < upper_band.iloc[i - 1] or df["Close"].iloc[i - 1] > upper_band.iloc[i - 1]:
            pass
        else:
            upper_band.iloc[i] = upper_band.iloc[i - 1]

        if supertrend.iloc[i - 1] == upper_band.iloc[i - 1]:
            if df["Close"].iloc[i] > upper_band.iloc[i]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
        else:
            if df["Close"].iloc[i] < lower_band.iloc[i]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1

    return float(supertrend.iloc[-1]), int(direction.iloc[-1])


# ============================================================
# CCI — Manual implementation
# ============================================================

def compute_cci(df: pd.DataFrame, period: int = 20):
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mean_dev = typical_price.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    cci = (typical_price - sma_tp) / (0.015 * mean_dev)
    return float(cci.iloc[-1])


# ============================================================
# PIVOT POINTS — Multiple methods
# ============================================================

def compute_pivot_points(high, low, close, method="standard"):
    """Compute pivot levels using different methods."""
    if method == "standard":
        pp = (high + low + close) / 3
        r1 = 2 * pp - low
        s1 = 2 * pp - high
        r2 = pp + (high - low)
        s2 = pp - (high - low)
        r3 = high + 2 * (pp - low)
        s3 = low - 2 * (high - pp)

    elif method == "fibonacci":
        pp = (high + low + close) / 3
        diff = high - low
        r1 = pp + 0.382 * diff
        r2 = pp + 0.618 * diff
        r3 = pp + 1.000 * diff
        s1 = pp - 0.382 * diff
        s2 = pp - 0.618 * diff
        s3 = pp - 1.000 * diff

    elif method == "camarilla":
        pp = (high + low + close) / 3
        diff = high - low
        r1 = close + diff * 1.1 / 12
        r2 = close + diff * 1.1 / 6
        r3 = close + diff * 1.1 / 4
        s1 = close - diff * 1.1 / 12
        s2 = close - diff * 1.1 / 6
        s3 = close - diff * 1.1 / 4

    elif method == "woodie":
        pp = (high + low + 2 * close) / 4
        r1 = 2 * pp - low
        s1 = 2 * pp - high
        r2 = pp + (high - low)
        s2 = pp - (high - low)
        r3 = high + 2 * (pp - low)
        s3 = low - 2 * (high - pp)

    else:
        # Fallback to standard
        return compute_pivot_points(high, low, close, "standard")

    return {"pp": pp, "r1": r1, "r2": r2, "r3": r3, "s1": s1, "s2": s2, "s3": s3}


# ============================================================
# INDICATOR SERVICE
# ============================================================

class IndicatorService:
    """
    Computes technical indicators with user-configurable parameters.
    """

    @staticmethod
    def compute_all(
        df: pd.DataFrame,
        current_price: float,
        stock_info: dict,
        config: dict = None,
    ) -> dict:
        """
        Compute all indicators with optional custom config.

        Args:
            df: OHLCV DataFrame
            current_price: Latest close price
            stock_info: Dict with high, low, prev_close
            config: Optional user config {indicator_id: {param: value}}
                    If None, uses all defaults.

        Returns:
            Dict mapping indicator_id -> ZoneResult dict
        """
        # Merge user config with defaults
        cfg = get_config_with_overrides(config)

        results = {}

        computations = {
            "rsi": lambda: IndicatorService._compute_rsi(df, current_price, cfg["rsi"]),
            "bollinger": lambda: IndicatorService._compute_bollinger(df, current_price, cfg["bollinger"]),
            "macd": lambda: IndicatorService._compute_macd(df, current_price, cfg["macd"]),
            "pivot": lambda: IndicatorService._compute_pivot(stock_info, current_price, cfg["pivot"]),
            "supertrend": lambda: IndicatorService._compute_supertrend(df, current_price, cfg["supertrend"]),
            "stochastic": lambda: IndicatorService._compute_stochastic(df, current_price, cfg["stochastic"]),
            "adx": lambda: IndicatorService._compute_adx(df, current_price, cfg["adx"]),
            "cci": lambda: IndicatorService._compute_cci(df, current_price, cfg["cci"]),
            "vwap": lambda: IndicatorService._compute_vwap(df, current_price, cfg["vwap"]),
            "ma_cross": lambda: IndicatorService._compute_ma_cross(df, current_price, cfg["ma_cross"]),
            "williams_r": lambda: IndicatorService._compute_williams_r(df, current_price, cfg["williams_r"]),
            "atr": lambda: IndicatorService._compute_atr(df, current_price, cfg["atr"]),
            "ema_cross": lambda: IndicatorService._compute_ema_cross(df, current_price, cfg["ema_cross"]),
            "obv": lambda: IndicatorService._compute_obv(df, current_price, cfg["obv"]),
        }

        for ind_id, compute_fn in computations.items():
            try:
                results[ind_id] = compute_fn()
            except Exception as e:
                logger.warning(f"{ind_id} computation failed: {e}")

        return results

    # --------------------------------------------------------
    # INDIVIDUAL INDICATORS (now accept config params)
    # --------------------------------------------------------

    @staticmethod
    def _compute_rsi(df, price, cfg):
        period = cfg["period"]
        ob = cfg["overbought"]
        os_ = cfg["oversold"]

        indicator = RSIIndicator(close=df["Close"], window=period)
        rsi = float(indicator.rsi().iloc[-1])

        if rsi >= ob:
            zone, signal, color, score = "Overbought", "SELL", "#ef4444", -2
        elif rsi >= ob - 10:
            zone, signal, color, score = "Bullish", "HOLD", "#f59e0b", 0
        elif rsi <= os_:
            zone, signal, color, score = "Oversold", "BUY", "#22c55e", 2
        elif rsi <= os_ + 10:
            zone, signal, color, score = "Bearish", "WATCH", "#f59e0b", 0
        else:
            zone, signal, color, score = "Neutral", "HOLD", "#94a3b8", 0

        return {
            "zone": zone, "signal": signal, "color": color, "score": score,
            "detail": f"RSI({period}) at {rsi:.1f} — {zone.lower()} zone (OB:{ob}/OS:{os_})",
            "value": f"{rsi:.1f}",
            "config_used": {"period": period, "overbought": ob, "oversold": os_},
            "raw": {"rsi": round(rsi, 2)},
        }

    @staticmethod
    def _compute_bollinger(df, price, cfg):
        period = cfg["period"]
        std_dev = cfg["std_dev"]

        bb = BollingerBands(close=df["Close"], window=period, window_dev=std_dev)
        upper = float(bb.bollinger_hband().iloc[-1])
        middle = float(bb.bollinger_mavg().iloc[-1])
        lower = float(bb.bollinger_lband().iloc[-1])

        if price >= upper:
            zone, signal, color, score = "Upper Band", "SELL", "#ef4444", -2
            detail = f"Price ₹{price:.0f} above upper band ₹{upper:.0f}"
        elif price <= lower:
            zone, signal, color, score = "Lower Band", "BUY", "#22c55e", 2
            detail = f"Price ₹{price:.0f} below lower band ₹{lower:.0f}"
        elif price > middle:
            zone, signal, color, score = "Above Middle", "BULLISH", "#86efac", 1
            detail = f"Price above middle band ₹{middle:.0f}"
        else:
            zone, signal, color, score = "Below Middle", "BEARISH", "#fbbf24", -1
            detail = f"Price below middle band ₹{middle:.0f}"

        return {
            "zone": zone, "signal": signal, "color": color, "score": score,
            "detail": f"BB({period},{std_dev}) — {detail}",
            "value": f"U:₹{upper:.0f} M:₹{middle:.0f} L:₹{lower:.0f}",
            "config_used": {"period": period, "std_dev": std_dev},
            "raw": {"upper": round(upper, 2), "middle": round(middle, 2), "lower": round(lower, 2)},
        }

    @staticmethod
    def _compute_macd(df, price, cfg):
        fast = cfg["fast_period"]
        slow = cfg["slow_period"]
        sig = cfg["signal_period"]

        macd_ind = MACD(close=df["Close"], window_fast=fast, window_slow=slow, window_sign=sig)
        macd_line = float(macd_ind.macd().iloc[-1])
        signal_line = float(macd_ind.macd_signal().iloc[-1])
        histogram = float(macd_ind.macd_diff().iloc[-1])

        if histogram > 0 and macd_line > 0:
            zone, signal, color, score = "Strong Bullish", "BUY", "#22c55e", 2
        elif histogram > 0:
            zone, signal, color, score = "Bullish Crossover", "BUY", "#86efac", 1
        elif histogram < 0 and macd_line < 0:
            zone, signal, color, score = "Strong Bearish", "SELL", "#ef4444", -2
        elif histogram < 0:
            zone, signal, color, score = "Bearish Crossover", "SELL", "#fbbf24", -1
        else:
            zone, signal, color, score = "Neutral", "HOLD", "#94a3b8", 0

        return {
            "zone": zone, "signal": signal, "color": color, "score": score,
            "detail": f"MACD({fast},{slow},{sig}) — Line: {macd_line:.2f}, Signal: {signal_line:.2f}, Hist: {histogram:.2f}",
            "value": f"Hist: {histogram:.2f}",
            "config_used": {"fast_period": fast, "slow_period": slow, "signal_period": sig},
            "raw": {"macd_line": round(macd_line, 2), "signal_line": round(signal_line, 2), "histogram": round(histogram, 2)},
        }

    @staticmethod
    def _compute_pivot(stock_info, price, cfg):
        method = cfg["method"]
        h, l, c = stock_info["high"], stock_info["low"], stock_info["prev_close"]

        levels = compute_pivot_points(h, l, c, method)
        pp, r1, r2, r3, s1, s2, s3 = levels["pp"], levels["r1"], levels["r2"], levels["r3"], levels["s1"], levels["s2"], levels["s3"]

        if price > r2:
            zone, signal, color, score = "Above R2", "STRONG SELL", "#ef4444", -2
        elif price > r1:
            zone, signal, color, score = "R1-R2 Zone", "SELL", "#f97316", -1
        elif price > pp:
            zone, signal, color, score = "Above Pivot", "BULLISH", "#86efac", 1
        elif price > s1:
            zone, signal, color, score = "Below Pivot", "BEARISH", "#fbbf24", -1
        elif price > s2:
            zone, signal, color, score = "S1-S2 Zone", "BUY", "#86efac", 1
        else:
            zone, signal, color, score = "Below S2", "STRONG BUY", "#22c55e", 2

        return {
            "zone": zone, "signal": signal, "color": color, "score": score,
            "detail": f"Pivot ({method.title()}) PP: ₹{pp:.0f} | R1: ₹{r1:.0f} R2: ₹{r2:.0f} | S1: ₹{s1:.0f} S2: ₹{s2:.0f}",
            "value": f"PP: ₹{pp:.0f} ({method})",
            "config_used": {"method": method},
            "raw": {"pp": round(pp, 2), "r1": round(r1, 2), "r2": round(r2, 2), "r3": round(r3, 2), "s1": round(s1, 2), "s2": round(s2, 2), "s3": round(s3, 2)},
        }

    @staticmethod
    def _compute_supertrend(df, price, cfg):
        period = cfg["period"]
        mult = cfg["multiplier"]
        st_value, st_direction = compute_supertrend(df, period=period, multiplier=mult)

        if st_direction == 1:
            return {
                "zone": "Bullish", "signal": "BUY", "color": "#22c55e", "score": 2,
                "detail": f"Supertrend({period},{mult}) ₹{st_value:.0f} — uptrend active",
                "value": f"₹{st_value:.0f}",
                "config_used": {"period": period, "multiplier": mult},
                "raw": {"supertrend": round(st_value, 2), "direction": "bullish"},
            }
        return {
            "zone": "Bearish", "signal": "SELL", "color": "#ef4444", "score": -2,
            "detail": f"Supertrend({period},{mult}) ₹{st_value:.0f} — downtrend active",
            "value": f"₹{st_value:.0f}",
            "config_used": {"period": period, "multiplier": mult},
            "raw": {"supertrend": round(st_value, 2), "direction": "bearish"},
        }

    @staticmethod
    def _compute_stochastic(df, price, cfg):
        k_period = cfg["k_period"]
        d_period = cfg["d_period"]
        ob = cfg["overbought"]
        os_ = cfg["oversold"]

        stoch = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"], window=k_period, smooth_window=d_period)
        k = float(stoch.stoch().iloc[-1])
        d = float(stoch.stoch_signal().iloc[-1])

        if k > ob and d > ob:
            zone, signal, color, score = "Overbought", "SELL", "#ef4444", -2
        elif k < os_ and d < os_:
            zone, signal, color, score = "Oversold", "BUY", "#22c55e", 2
        elif k > d:
            zone, signal, color, score = "Bullish", "BUY", "#86efac", 1
        else:
            zone, signal, color, score = "Bearish", "SELL", "#fbbf24", -1

        return {
            "zone": zone, "signal": signal, "color": color, "score": score,
            "detail": f"Stoch({k_period},{d_period}) %K: {k:.1f}, %D: {d:.1f} (OB:{ob}/OS:{os_})",
            "value": f"%K: {k:.1f}",
            "config_used": {"k_period": k_period, "d_period": d_period, "overbought": ob, "oversold": os_},
            "raw": {"k": round(k, 2), "d": round(d, 2)},
        }

    @staticmethod
    def _compute_adx(df, price, cfg):
        period = cfg["period"]
        strong = cfg["strong_trend"]

        adx_ind = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=period)
        adx = float(adx_ind.adx().iloc[-1])
        plus_di = float(adx_ind.adx_pos().iloc[-1])
        minus_di = float(adx_ind.adx_neg().iloc[-1])

        if adx > strong * 2:
            zone, signal, color = "Strong Trend", "TRENDING", "#8b5cf6"
        elif adx > strong:
            zone, signal, color = "Trending", "TREND", "#a78bfa"
        else:
            zone, signal, color = "Weak/No Trend", "RANGING", "#94a3b8"

        direction = "bullish" if plus_di > minus_di else "bearish"

        return {
            "zone": zone, "signal": signal, "color": color, "score": 0,
            "detail": f"ADX({period}) at {adx:.1f} | +DI: {plus_di:.1f} | -DI: {minus_di:.1f} ({direction})",
            "value": f"{adx:.1f}",
            "config_used": {"period": period, "strong_trend": strong},
            "raw": {"adx": round(adx, 2), "plus_di": round(plus_di, 2), "minus_di": round(minus_di, 2)},
        }

    @staticmethod
    def _compute_cci(df, price, cfg):
        period = cfg["period"]
        ob = cfg["overbought"]
        os_ = cfg["oversold"]

        cci = compute_cci(df, period=period)

        if cci > ob * 2:
            zone, signal, color, score = "Extreme Overbought", "STRONG SELL", "#ef4444", -2
        elif cci > ob:
            zone, signal, color, score = "Overbought", "SELL", "#f97316", -1
        elif cci < os_ * 2:
            zone, signal, color, score = "Extreme Oversold", "STRONG BUY", "#22c55e", 2
        elif cci < os_:
            zone, signal, color, score = "Oversold", "BUY", "#86efac", 1
        else:
            zone, signal, color, score = "Neutral", "HOLD", "#94a3b8", 0

        return {
            "zone": zone, "signal": signal, "color": color, "score": score,
            "detail": f"CCI({period}) at {cci:.0f} (OB:{ob}/OS:{os_})",
            "value": f"{cci:.0f}",
            "config_used": {"period": period, "overbought": ob, "oversold": os_},
            "raw": {"cci": round(cci, 2)},
        }

    @staticmethod
    def _compute_vwap(df, price, cfg):
        threshold = cfg["threshold_pct"]

        try:
            vwap_ind = VolumeWeightedAveragePrice(high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"])
            vwap = float(vwap_ind.volume_weighted_average_price().iloc[-1])
        except Exception:
            typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
            vwap = float((typical_price * df["Volume"]).cumsum().iloc[-1] / df["Volume"].cumsum().iloc[-1])

        pct_diff = ((price - vwap) / vwap) * 100

        if pct_diff > threshold:
            zone, signal, color, score = "Above VWAP", "BULLISH", "#22c55e", 1
        elif pct_diff > 0:
            zone, signal, color, score = "Near Above VWAP", "HOLD", "#86efac", 0
        elif pct_diff < -threshold:
            zone, signal, color, score = "Below VWAP", "BEARISH", "#ef4444", -1
        else:
            zone, signal, color, score = "Near Below VWAP", "HOLD", "#fbbf24", 0

        return {
            "zone": zone, "signal": signal, "color": color, "score": score,
            "detail": f"VWAP: ₹{vwap:.0f} (price {pct_diff:+.1f}% | threshold: ±{threshold}%)",
            "value": f"₹{vwap:.0f}",
            "config_used": {"threshold_pct": threshold},
            "raw": {"vwap": round(vwap, 2), "pct_diff": round(pct_diff, 2)},
        }

    @staticmethod
    def _compute_ma_cross(df, price, cfg):
        fast = cfg["fast_period"]
        slow = cfg["slow_period"]

        sma_fast = SMAIndicator(close=df["Close"], window=fast).sma_indicator()
        sma_slow = SMAIndicator(close=df["Close"], window=slow).sma_indicator()

        fast_val = sma_fast.iloc[-1]
        slow_val = sma_slow.iloc[-1]

        if pd.isna(slow_val):
            if not pd.isna(fast_val):
                fv = float(fast_val)
                sig = "BULLISH" if price > fv else "BEARISH"
                return {
                    "zone": f"{'Above' if price > fv else 'Below'} SMA{fast}", "signal": sig,
                    "color": "#86efac" if price > fv else "#fbbf24", "score": 1 if price > fv else -1,
                    "detail": f"SMA{fast}: ₹{fv:.0f} (SMA{slow} needs more data)",
                    "value": f"SMA{fast}: ₹{fv:.0f}",
                    "config_used": {"fast_period": fast, "slow_period": slow},
                    "raw": {"sma_fast": round(fv, 2)},
                }
            return {
                "zone": "Insufficient Data", "signal": "HOLD", "color": "#94a3b8", "score": 0,
                "detail": f"Need {slow}+ days of data", "value": "N/A",
                "config_used": {"fast_period": fast, "slow_period": slow}, "raw": {},
            }

        fv, sv = float(fast_val), float(slow_val)

        if price > fv and fv > sv:
            zone, signal, color, score = "Golden Cross", "STRONG BUY", "#22c55e", 2
        elif price < fv and fv < sv:
            zone, signal, color, score = "Death Cross", "STRONG SELL", "#ef4444", -2
        elif price > fv:
            zone, signal, color, score = f"Above SMA{fast}", "BULLISH", "#86efac", 1
        else:
            zone, signal, color, score = f"Below SMA{fast}", "BEARISH", "#fbbf24", -1

        return {
            "zone": zone, "signal": signal, "color": color, "score": score,
            "detail": f"SMA{fast}: ₹{fv:.0f} | SMA{slow}: ₹{sv:.0f}",
            "value": f"SMA{fast}: ₹{fv:.0f}",
            "config_used": {"fast_period": fast, "slow_period": slow},
            "raw": {"sma_fast": round(fv, 2), "sma_slow": round(sv, 2)},
        }

    @staticmethod
    def _compute_williams_r(df, price, cfg):
        period = cfg["period"]
        ob = cfg["overbought"]
        os_ = cfg["oversold"]

        wr_ind = WilliamsRIndicator(high=df["High"], low=df["Low"], close=df["Close"], lbp=period)
        val = float(wr_ind.williams_r().iloc[-1])

        if val > ob:
            zone, signal, color, score = "Overbought", "SELL", "#ef4444", -2
        elif val < os_:
            zone, signal, color, score = "Oversold", "BUY", "#22c55e", 2
        elif val > -50:
            zone, signal, color, score = "Upper Half", "HOLD", "#f59e0b", 0
        else:
            zone, signal, color, score = "Lower Half", "WATCH", "#fbbf24", 0

        return {
            "zone": zone, "signal": signal, "color": color, "score": score,
            "detail": f"Williams %R({period}) at {val:.1f} (OB:{ob}/OS:{os_})",
            "value": f"{val:.1f}",
            "config_used": {"period": period, "overbought": ob, "oversold": os_},
            "raw": {"williams_r": round(val, 2)},
        }

    @staticmethod
    def _compute_atr(df, price, cfg):
        period = cfg["period"]
        high_vol = cfg["high_volatility_pct"]

        atr_ind = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=period)
        val = float(atr_ind.average_true_range().iloc[-1])
        pct = (val / price) * 100

        if pct > high_vol:
            zone, signal, color = "High Volatility", "VOLATILE", "#ef4444"
        elif pct > high_vol / 2:
            zone, signal, color = "Moderate Volatility", "NORMAL", "#f59e0b"
        else:
            zone, signal, color = "Low Volatility", "CALM", "#22c55e"

        return {
            "zone": zone, "signal": signal, "color": color, "score": 0,
            "detail": f"ATR({period}): ₹{val:.2f} ({pct:.1f}% of price)",
            "value": f"₹{val:.2f}",
            "config_used": {"period": period, "high_volatility_pct": high_vol},
            "raw": {"atr": round(val, 2), "atr_pct": round(pct, 2)},
        }

    @staticmethod
    def _compute_ema_cross(df, price, cfg):
        fast = cfg["fast_period"]
        slow = cfg["slow_period"]

        ema_fast = float(EMAIndicator(close=df["Close"], window=fast).ema_indicator().iloc[-1])
        ema_slow = float(EMAIndicator(close=df["Close"], window=slow).ema_indicator().iloc[-1])

        if price > ema_fast and ema_fast > ema_slow:
            zone, signal, color, score = "Bullish Cross", "BUY", "#22c55e", 2
        elif price < ema_fast and ema_fast < ema_slow:
            zone, signal, color, score = "Bearish Cross", "SELL", "#ef4444", -2
        elif ema_fast > ema_slow:
            zone, signal, color, score = f"EMA{fast} Above", "BULLISH", "#86efac", 1
        else:
            zone, signal, color, score = f"EMA{fast} Below", "BEARISH", "#fbbf24", -1

        return {
            "zone": zone, "signal": signal, "color": color, "score": score,
            "detail": f"EMA{fast}: ₹{ema_fast:.0f} | EMA{slow}: ₹{ema_slow:.0f}",
            "value": f"EMA{fast}: ₹{ema_fast:.0f}",
            "config_used": {"fast_period": fast, "slow_period": slow},
            "raw": {"ema_fast": round(ema_fast, 2), "ema_slow": round(ema_slow, 2)},
        }

    @staticmethod
    def _compute_obv(df, price, cfg):
        lookback = cfg["lookback_days"]

        obv_ind = OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"])
        obv_series = obv_ind.on_balance_volume()
        obv_current = float(obv_series.iloc[-1])
        obv_prev = float(obv_series.iloc[-lookback - 1]) if len(obv_series) > lookback else float(obv_series.iloc[0])

        obv_change = obv_current - obv_prev
        obv_display = f"{obv_current / 1e6:.1f}M"

        price_prev = float(df["Close"].iloc[-lookback - 1]) if len(df) > lookback else float(df["Close"].iloc[0])

        if obv_change > 0 and price > price_prev:
            zone, signal, color, score = "Accumulation", "BULLISH", "#22c55e", 1
        elif obv_change < 0 and price < price_prev:
            zone, signal, color, score = "Distribution", "BEARISH", "#ef4444", -1
        elif obv_change > 0:
            zone, signal, color, score = "OBV Rising", "WATCH", "#86efac", 0
        else:
            zone, signal, color, score = "OBV Falling", "WATCH", "#fbbf24", 0

        return {
            "zone": zone, "signal": signal, "color": color, "score": score,
            "detail": f"OBV: {obv_display} ({lookback}-day change: {obv_change / 1e6:+.1f}M)",
            "value": obv_display,
            "config_used": {"lookback_days": lookback},
            "raw": {"obv": round(obv_current, 0), "obv_change": round(obv_change, 0)},
        }
