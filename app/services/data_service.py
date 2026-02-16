"""
DataService — loads pre-fetched stock data from stock_data.json
No API calls at startup = instant loading on Render.
Data is fetched daily using fetch_daily_data.py and committed to git.
"""

import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Optional
import asyncio
import logging

from app.config import NIFTY50_STOCKS

logger = logging.getLogger(__name__)

DATA_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "stock_data.json"
)


class DataService:
    def __init__(self):
        self.is_initialized = False
        self._cache: dict[str, dict] = {}
        self._ohlcv_cache: dict[str, pd.DataFrame] = {}
        self._last_fetch: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=5)
        self._custom_symbols: set = set()
        self._data_date: str = ""

    async def initialize(self):
        try:
            await self.refresh_all()
            self.is_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            self.is_initialized = False

    async def refresh_all(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_from_file)
        self._last_fetch = datetime.now()

    def _load_from_file(self):
        """Load stock data from stock_data.json"""

        possible_paths = [
            DATA_FILE,
            "stock_data.json",
            os.path.join(os.getcwd(), "stock_data.json"),
            "/app/stock_data.json",  # Docker path
            "/opt/render/project/src/stock_data.json",  # Native path
        ]

        data_path = None
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break

        if not data_path:
            logger.error("❌ stock_data.json not found!")
            return

        try:
            with open(data_path, "r") as f:
                raw = json.load(f)

            self._data_date = raw.get("fetched_date", "unknown")
            stocks_data = raw.get("stocks", {})

            logger.info(f"📂 Loading data from {data_path}")
            logger.info(f"📅 Data date: {self._data_date}")
            logger.info(f"📊 Stocks in file: {len(stocks_data)}")

            success = 0

            for symbol, stock_info in stocks_data.items():
                try:
                    candles = stock_info.get("candles", [])

                    if not candles:
                        continue

                    # Filter valid candle rows (must have 6 values)
                    candles = [c for c in candles if isinstance(c, list) and len(c) == 6]

                    if not candles:
                        continue

                    df = pd.DataFrame(
                        candles,
                        columns=["Date", "Open", "High", "Low", "Close", "Volume"]
                    )

                    df["Date"] = pd.to_datetime(df["Date"], unit="s")

                    for col in ["Open", "High", "Low", "Close", "Volume"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                    df = df.dropna()
                    df = df.sort_values("Date").reset_index(drop=True)

                    if df.empty:
                        continue

                    name = stock_info.get("name", symbol)
                    sector = stock_info.get("sector", "Other")

                    self._ohlcv_cache[symbol] = df.copy()
                    self._process_stock_data(symbol, df, name, sector)

                    success += 1

                except Exception as e:
                    logger.error(f"❌ Failed processing {symbol}: {e}")

            logger.info(f"📦 Final cache size: {len(self._cache)}")
            logger.info(f"✅ {success}/{len(stocks_data)} stocks loaded!")

        except Exception as e:
            logger.error(f"❌ Failed to load stock_data.json: {e}")

    def _process_stock_data(self, symbol: str, df: pd.DataFrame, name: str, sector: str):
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]

        high_52w = df["High"].tail(252).max() if len(df) >= 252 else df["High"].max()
        low_52w = df["Low"].tail(252).min() if len(df) >= 252 else df["Low"].min()

        volume = int(latest["Volume"]) if pd.notna(latest.get("Volume")) else 0

        self._cache[symbol] = {
            "symbol": symbol,
            "name": name,
            "sector": sector,
            "exchange": "NSE",
            "price": round(float(latest["Close"]), 2),
            "change": round(float(latest["Close"] - prev["Close"]), 2),
            "change_pct": round(
                float((latest["Close"] - prev["Close"]) / prev["Close"] * 100), 2
            ),
            "prev_close": round(float(prev["Close"]), 2),
            "open": round(float(latest["Open"]), 2),
            "high": round(float(latest["High"]), 2),
            "low": round(float(latest["Low"]), 2),
            "volume": volume,
            "high_52w": round(float(high_52w), 2),
            "low_52w": round(float(low_52w), 2),
            "is_custom": symbol in self._custom_symbols,
            "data_date": self._data_date,
        }

    def get_stock(self, symbol: str) -> Optional[dict]:
        return self._cache.get(symbol.upper())

    def get_all_stocks(self) -> list[dict]:
        return list(self._cache.values())

    def get_ohlcv(self, symbol: str) -> Optional[pd.DataFrame]:
        return self._ohlcv_cache.get(symbol.upper())

    def get_sectors(self) -> list[str]:
        return sorted(list(set(s["sector"] for s in self._cache.values())))

    @property
    def needs_refresh(self) -> bool:
        return False
data_service = DataService()