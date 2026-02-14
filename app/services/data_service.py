"""
DataService — fetches and caches stock data from Yahoo Finance.
Supports preloaded NIFTY 50 stocks AND on-demand fetching of any NSE stock.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import asyncio
import logging

from app.config import NIFTY50_STOCKS

logger = logging.getLogger(__name__)


class DataService:
    def __init__(self):
        self.is_initialized = False
        self._cache: dict[str, dict] = {}
        self._ohlcv_cache: dict[str, pd.DataFrame] = {}
        self._last_fetch: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=5)
        # Track which symbols were added on-demand (not in NIFTY50)
        self._custom_symbols: set = set()

    async def initialize(self):
        try:
            await self.refresh_all()
            self.is_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            self.is_initialized = False

    async def refresh_all(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._fetch_all_sync)
        self._last_fetch = datetime.now()

    def _fetch_all_sync(self):
        symbols = [info["yahoo"] for info in NIFTY50_STOCKS.values()]
        logger.info(f"Fetching data for {len(symbols)} stocks...")

        try:
            data = yf.download(
                tickers=symbols,
                period="6mo",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
            )
        except Exception as e:
            logger.error(f"yfinance download failed: {e}")
            self._fetch_individual(symbols)
            return

        for nse_symbol, info in NIFTY50_STOCKS.items():
            yahoo_symbol = info["yahoo"]
            try:
                if len(symbols) == 1:
                    df = data
                else:
                    df = data[yahoo_symbol].dropna()

                if df.empty:
                    logger.warning(f"No data for {nse_symbol}")
                    continue

                self._ohlcv_cache[nse_symbol] = df.copy()
                self._process_stock_data(nse_symbol, df, info["name"], info["sector"])

            except Exception as e:
                logger.warning(f"Error processing {nse_symbol}: {e}")

    def _fetch_individual(self, symbols):
        for nse_symbol, info in NIFTY50_STOCKS.items():
            try:
                ticker = yf.Ticker(info["yahoo"])
                df = ticker.history(period="6mo", interval="1d")
                if not df.empty:
                    self._ohlcv_cache[nse_symbol] = df.copy()
                    self._process_stock_data(nse_symbol, df, info["name"], info["sector"])
            except Exception as e:
                logger.warning(f"Individual fetch failed for {nse_symbol}: {e}")

    def _process_stock_data(self, symbol: str, df: pd.DataFrame, name: str, sector: str):
        """Process OHLCV DataFrame into stock info dict."""
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]

        high_52w = df["High"].tail(252).max() if len(df) >= 252 else df["High"].max()
        low_52w = df["Low"].tail(252).min() if len(df) >= 252 else df["Low"].min()

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
            "volume": int(latest["Volume"]),
            "high_52w": round(float(high_52w), 2),
            "low_52w": round(float(low_52w), 2),
            "is_custom": symbol in self._custom_symbols,
        }

    # ============================================================
    # ON-DEMAND FETCHING — Any NSE stock
    # ============================================================

    def fetch_on_demand_sync(self, symbol: str) -> Optional[dict]:
        """
        Fetch any NSE stock on-demand by its symbol.
        Appends .NS for Yahoo Finance and fetches 6 months of data.
        Returns the stock info dict or None if not found.
        """
        symbol = symbol.upper().strip()

        # Already cached and fresh? Return it
        if symbol in self._cache:
            return self._cache[symbol]

        yahoo_symbol = f"{symbol}.NS"
        logger.info(f"On-demand fetch: {symbol} ({yahoo_symbol})")

        try:
            ticker = yf.Ticker(yahoo_symbol)

            # Get stock info for name and sector
            try:
                info = ticker.info
                name = info.get("shortName") or info.get("longName") or symbol
                sector = info.get("sector") or "Other"
                # Clean up sector name
                if sector == "Financial Services":
                    sector = "Banking"
                elif sector == "Information Technology":
                    sector = "IT"
                elif sector == "Consumer Defensive":
                    sector = "FMCG"
                elif sector == "Consumer Cyclical":
                    sector = "Consumer"
                elif sector == "Basic Materials":
                    sector = "Metals"
                elif sector == "Communication Services":
                    sector = "Telecom"
            except Exception:
                name = symbol
                sector = "Other"

            # Get historical data
            df = ticker.history(period="6mo", interval="1d")

            if df.empty:
                logger.warning(f"No data found for {yahoo_symbol}")
                return None

            # Standardize column names (yfinance history uses title case)
            df.columns = [c.title() if c[0].islower() else c for c in df.columns]

            self._ohlcv_cache[symbol] = df.copy()
            self._custom_symbols.add(symbol)
            self._process_stock_data(symbol, df, name, sector)

            logger.info(f"✅ On-demand loaded: {symbol} ({name}) ₹{self._cache[symbol]['price']}")
            return self._cache[symbol]

        except Exception as e:
            logger.error(f"On-demand fetch failed for {symbol}: {e}")
            return None

    async def fetch_on_demand(self, symbol: str) -> Optional[dict]:
        """Async wrapper for on-demand fetching."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.fetch_on_demand_sync, symbol)

    def remove_custom_stock(self, symbol: str):
        """Remove a custom stock from cache."""
        symbol = symbol.upper()
        if symbol in self._custom_symbols:
            self._custom_symbols.discard(symbol)
            self._cache.pop(symbol, None)
            self._ohlcv_cache.pop(symbol, None)
            return True
        return False

    # ============================================================
    # EXISTING GETTERS
    # ============================================================

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
        if self._last_fetch is None:
            return True
        return datetime.now() - self._last_fetch > self._cache_ttl
