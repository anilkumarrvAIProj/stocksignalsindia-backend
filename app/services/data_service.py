"""
DataService — loads pre-fetched stock data from stock_data.json
No API calls at startup = instant loading on Render.
Data is fetched daily using fetch_daily_data.py and committed to git.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Optional
import asyncio
import logging

from app.config import NIFTY50_STOCKS

logger = logging.getLogger(__name__)

# Path to the pre-fetched data file
DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "stock_data.json")


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
        """Load stock data from pre-fetched JSON file. Instant startup."""
        
        # Try multiple paths
        possible_paths = [
            DATA_FILE,
            "stock_data.json",
            os.path.join(os.getcwd(), "stock_data.json"),
            "/opt/render/project/src/stock_data.json",
        ]
        
        data_path = None
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        
        if not data_path:
            logger.error(f"❌ stock_data.json not found! Run fetch_daily_data.py first.")
            logger.error(f"   Searched: {possible_paths}")
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
                    
                    # Convert candles to DataFrame
                    # candles: [[timestamp, open, high, low, close, volume], ...]
                    df = pd.DataFrame(candles, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
                    df["Date"] = pd.to_datetime(df["Date"], unit="s")
                    df = df.sort_values("Date").reset_index(drop=True)
                    
                    name = stock_info.get("name", symbol)
                    sector = stock_info.get("sector", "Other")
                    
                    self._ohlcv_cache[symbol] = df.copy()
                    self._process_stock_data(symbol, df, name, sector)
                    success += 1
                    
                except Exception as e:
                    logger.warning(f"❌ Error processing {symbol}: {e}")
            
            logger.info(f"✅ {success}/{len(stocks_data)} stocks loaded instantly from file!")
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in stock_data.json: {e}")
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

    # ============================================================
    # ON-DEMAND FETCHING (uses Fyers API for stocks not in file)
    # ============================================================

    def _init_fyers(self):
        """Initialize Fyers client for on-demand fetches."""
        try:
            from fyers_apiv3 import fyersModel
            client_id = os.environ.get("FYERS_CLIENT_ID", "")
            access_token = os.environ.get("FYERS_ACCESS_TOKEN", "")
            if client_id and access_token:
                self._fyers = fyersModel.FyersModel(
                    client_id=client_id, token=access_token,
                    is_async=False, log_path=""
                )
                logger.info("✅ Fyers API available for on-demand stock lookups")
            else:
                self._fyers = None
                logger.info("ℹ Fyers credentials not set — on-demand lookups disabled")
        except Exception as e:
            self._fyers = None
            logger.warning(f"⚠ Fyers init failed: {e}")

    def fetch_on_demand_sync(self, symbol: str) -> Optional[dict]:
        """Fetch any NSE stock — from cache or Fyers API."""
        symbol = symbol.upper().strip()
        if symbol in self._cache:
            return self._cache[symbol]

        # Try Fyers API
        if not hasattr(self, '_fyers') or self._fyers is None:
            self._init_fyers()

        if not hasattr(self, '_fyers') or self._fyers is None:
            return None

        logger.info(f"On-demand fetch via Fyers: {symbol}")
        try:
            from datetime import date
            fyers_symbol = f"NSE:{symbol}-EQ"
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

            data = {
                "symbol": fyers_symbol,
                "resolution": "D",
                "date_format": "1",
                "range_from": start_date,
                "range_to": end_date,
                "cont_flag": "1"
            }

            response = self._fyers.history(data=data)

            if response.get("s") != "ok" or not response.get("candles"):
                logger.warning(f"❌ Fyers returned no data for {symbol}: {response.get('message', '')}")
                return None

            df = pd.DataFrame(response["candles"], columns=["Date", "Open", "High", "Low", "Close", "Volume"])
            df["Date"] = pd.to_datetime(df["Date"], unit="s")
            df = df.sort_values("Date").reset_index(drop=True)

            name = symbol
            sector = "Other"

            self._ohlcv_cache[symbol] = df.copy()
            self._custom_symbols.add(symbol)
            self._process_stock_data(symbol, df, name, sector)
            logger.info(f"✅ On-demand: {symbol} ₹{self._cache[symbol]['price']}")
            return self._cache[symbol]

        except Exception as e:
            logger.error(f"On-demand failed {symbol}: {e}")
            return None

    async def fetch_on_demand(self, symbol: str) -> Optional[dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.fetch_on_demand_sync, symbol)

    def remove_custom_stock(self, symbol: str):
        symbol = symbol.upper()
        if symbol in self._custom_symbols:
            self._custom_symbols.discard(symbol)
            self._cache.pop(symbol, None)
            self._ohlcv_cache.pop(symbol, None)
            return True
        return False

    def get_stock(self, symbol: str) -> Optional[dict]:
        return self._cache.get(symbol.upper())

    def get_all_stocks(self) -> list[dict]:
        return list(self._cache.values())

    def get_ohlcv(self, symbol: str) -> Optional[pd.DataFrame]:
        return self._ohlcv_cache.get(symbol.upper())

    def get_sectors(self) -> list[str]:
        return sorted(list(set(s["sector"] for s in self._cache.values())))

    def get_custom_symbols(self) -> list[str]:
        return sorted(list(self._custom_symbols))

    @property
    def needs_refresh(self) -> bool:
        # No refresh needed — data comes from file
        return False
data_service = DataService()