"""
Pydantic models for API request/response schemas.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ============================================================
# ENUMS
# ============================================================

class SignalType(str, Enum):
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    BULLISH = "BULLISH"
    HOLD = "HOLD"
    WATCH = "WATCH"
    BEARISH = "BEARISH"
    SELL = "SELL"
    STRONG_SELL = "STRONG SELL"
    TRENDING = "TRENDING"
    TREND = "TREND"
    RANGING = "RANGING"


class IndicatorCategory(str, Enum):
    MOMENTUM = "Momentum"
    VOLATILITY = "Volatility"
    TREND = "Trend"
    VOLUME = "Volume"
    SUPPORT_RESIST = "Support/Resistance"


class IndicatorId(str, Enum):
    RSI = "rsi"
    BOLLINGER = "bollinger"
    MACD = "macd"
    PIVOT = "pivot"
    SUPERTREND = "supertrend"
    STOCHASTIC = "stochastic"
    ADX = "adx"
    CCI = "cci"
    VWAP = "vwap"
    MA_CROSS = "ma_cross"
    WILLIAMS_R = "williams_r"
    ATR = "atr"
    EMA_CROSS = "ema_cross"
    OBV = "obv"
    ICHIMOKU = "ichimoku"


# ============================================================
# RESPONSE MODELS
# ============================================================

class ZoneResult(BaseModel):
    """Result of a zone analysis for a single indicator."""
    zone: str = Field(..., description="Zone name, e.g. 'Overbought', 'Lower Band'")
    signal: str = Field(..., description="Trading signal: BUY, SELL, HOLD, etc.")
    color: str = Field(..., description="Hex color for UI rendering")
    score: int = Field(..., description="Numeric score: -2 (strong sell) to +2 (strong buy)")
    detail: str = Field(..., description="Human-readable explanation")
    value: str = Field(..., description="Display value of the indicator")
    education: str = Field("", description="Educational tooltip about the indicator")


class IndicatorZone(BaseModel):
    """Full indicator zone info including metadata."""
    id: str
    name: str
    full_name: str
    category: str
    zone: ZoneResult


class StockPrice(BaseModel):
    """Current price data for a stock."""
    symbol: str
    name: str
    sector: str
    exchange: str = "NSE"
    price: float
    change: float
    change_pct: float
    prev_close: float
    open: float
    high: float
    low: float
    volume: int
    high_52w: float
    low_52w: float
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None


class StockZoneSummary(BaseModel):
    """Summary of all zones for a stock — the main card view."""
    stock: StockPrice
    zones: list[IndicatorZone]
    consensus_score: int = Field(..., description="Sum of all zone scores")
    buy_count: int
    sell_count: int
    neutral_count: int
    consensus_label: str = Field(..., description="Strong Buy / Buy / Neutral / Sell / Strong Sell")


class StockListItem(BaseModel):
    """Compact stock item for the list/table view."""
    symbol: str
    name: str
    sector: str
    price: float
    change_pct: float
    consensus_score: int
    buy_count: int
    sell_count: int
    neutral_count: int
    consensus_label: str
    zone_summary: list[dict] = Field(
        ..., description="Compact zone info: [{id, signal, color}]"
    )


class ScannerRequest(BaseModel):
    """Request body for the stock scanner."""
    indicator: Optional[IndicatorId] = None
    signal: Optional[str] = None  # "BUY", "SELL", etc.
    min_score: Optional[int] = None
    max_score: Optional[int] = None
    sector: Optional[str] = None
    min_change_pct: Optional[float] = None
    max_change_pct: Optional[float] = None
    sort_by: str = "consensus_score"
    sort_order: str = "desc"
    limit: int = 50


class ScannerResponse(BaseModel):
    """Response from the stock scanner."""
    total: int
    results: list[StockListItem]
    filters_applied: dict
