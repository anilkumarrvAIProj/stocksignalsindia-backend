"""
DataService — fetches stock data directly from NSE India website.
No yfinance, no jugaad-data — just direct HTTP requests to nseindia.com.
Works on cloud servers (Render, Railway, etc.)
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, date
from typing import Optional
import asyncio
import logging
import time
import random
import json

from app.config import NIFTY50_STOCKS

logger = logging.getLogger(__name__)


class NSEFetcher:
    """Direct NSE India data fetcher using their internal API."""
    
    BASE_URL = "https://www.nseindia.com"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        })
        self._cookies_set = False
    
    def _set_cookies(self):
        """Visit NSE homepage first to get cookies (required by NSE)."""
        if self._cookies_set:
            return
        try:
            r = self.session.get(self.BASE_URL, timeout=10)
            self._cookies_set = True
            logger.info("🍪 NSE cookies obtained")
        except Exception as e:
            logger.warning(f"Cookie fetch failed: {e}")
    
    def get_stock_quote(self, symbol: str) -> Optional[dict]:
        """Get current quote for a stock from NSE."""
        self._set_cookies()
        try:
            url = f"{self.BASE_URL}/api/quote-equity?symbol={symbol}"
            r = self.session.get(url, timeout=10)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            logger.debug(f"Quote failed for {symbol}: {e}")
        return None
    
    def get_historical_data(self, symbol: str, days: int = 180) -> Optional[pd.DataFrame]:
        """Get historical OHLCV data from NSE."""
        self._set_cookies()
        
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        try:
            url = (
                f"{self.BASE_URL}/api/historical/cm/equity?"
                f"symbol={symbol}"
                f"&series=[%22EQ%22]"
                f"&from={start_date.strftime('%d-%m-%Y')}"
                f"&to={end_date.strftime('%d-%m-%Y')}"
            )
            r = self.session.get(url, timeout=15)
            
            if r.status_code != 200:
                return None
            
            data = r.json()
            
            if "data" not in data or not data["data"]:
                return None
            
            rows = []
            for item in data["data"]:
                try:
                    rows.append({
                        "Date": item.get("CH_TIMESTAMP", ""),
                        "Open": float(item.get("CH_OPENING_PRICE", 0)),
                        "High": float(item.get("CH_TRADE_HIGH_PRICE", 0)),
                        "Low": float(item.get("CH_TRADE_LOW_PRICE", 0)),
                        "Close": float(item.get("CH_CLOSING_PRICE", 0)),
                        "Volume": int(item.get("CH_TOT_TRADED_QTY", 0)),
                        "Prev_Close": float(item.get("CH_PREVIOUS_CLS_PRICE", 0)),
                    })
                except (ValueError, TypeError):
                    continue
            
            if not rows:
                return None
            
            df = pd.DataFrame(rows)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
            return df
            
        except Exception as e:
            logger.debug(f"Historical data failed for {symbol}: {e}")
            return None


class DataService:
    def __init__(self):
        self.is_initialized = False
        self._cache: dict[str, dict] = {}
        self._ohlcv_cache: dict[str, pd.DataFrame] = {}
        self._last_fetch: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=5)
        self._custom_symbols: set = set()
        self._nse = NSEFetcher()

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
        total = len(NIFTY50_STOCKS)
        logger.info(f"📊 Fetching {total} stocks from NSE India...")
        
        success_count = 0
        failed_symbols = []

        for i, (nse_symbol, info) in enumerate(NIFTY50_STOCKS.items()):
            try:
                if i > 0:
                    time.sleep(random.uniform(0.5, 1.5))

                # Re-establish cookies every 15 stocks
                if i % 15 == 0 and i > 0:
                    self._nse._cookies_set = False
                    self._nse.session.cookies.clear()
                    time.sleep(2)

                df = self._nse.get_historical_data(nse_symbol)

                if df is None or df.empty:
                    failed_symbols.append(nse_symbol)
                    if (i + 1) % 10 == 0:
                        logger.warning(f"[{i+1}/{total}] ❌ {nse_symbol} — no data")
                    continue

                self._ohlcv_cache[nse_symbol] = df.copy()
                self._process_stock_data(nse_symbol, df, info["name"], info["sector"])
                success_count += 1

                if success_count % 10 == 0 or success_count == 1:
                    logger.info(f"[{i+1}/{total}] ✅ {nse_symbol}: ₹{self._cache[nse_symbol]['price']} ({success_count} loaded)")

            except Exception as e:
                failed_symbols.append(nse_symbol)
                logger.warning(f"[{i+1}/{total}] ❌ {nse_symbol}: {str(e)[:80]}")
                time.sleep(1)

        logger.info(f"📊 First pass: {success_count}/{total} loaded")

        # Retry failed ones
        if failed_symbols:
            logger.info(f"🔄 Retrying {len(failed_symbols)} failed stocks...")
            self._nse._cookies_set = False
            self._nse.session.cookies.clear()
            time.sleep(3)
            
            retry_success = 0
            for nse_symbol in failed_symbols:
                info = NIFTY50_STOCKS.get(nse_symbol)
                if not info:
                    continue
                try:
                    time.sleep(random.uniform(1.5, 3))
                    df = self._nse.get_historical_data(nse_symbol)
                    if df is not None and not df.empty:
                        self._ohlcv_cache[nse_symbol] = df.copy()
                        self._process_stock_data(nse_symbol, df, info["name"], info["sector"])
                        retry_success += 1
                        logger.info(f"[Retry] ✅ {nse_symbol}")
                except Exception:
                    pass

            logger.info(f"📊 Retry: +{retry_success} loaded")

        logger.info(f"✅ TOTAL: {len(self._cache)}/{total} stocks ready")

    def _process_stock_data(self, symbol: str, df: pd.DataFrame, name: str, sector: str):
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]

        high_52w = df["High"].tail(252).max() if len(df) >= 252 else df["High"].max()
        low_52w = df["Low"].tail(252).min() if len(df) >= 252 else df["Low"].min()

        volume = int(latest["Volume"]) if "Volume" in df.columns and pd.notna(latest.get("Volume")) else 0

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
        }

    # ============================================================
    # ON-DEMAND FETCHING
    # ============================================================

    def fetch_on_demand_sync(self, symbol: str) -> Optional[dict]:
        symbol = symbol.upper().strip()
        if symbol in self._cache:
            return self._cache[symbol]

        logger.info(f"On-demand fetch: {symbol}")
        try:
            df = self._nse.get_historical_data(symbol)
            if df is None or df.empty:
                return None

            name = symbol
            sector = "Other"

            # Try to get name from quote
            try:
                quote = self._nse.get_stock_quote(symbol)
                if quote and "info" in quote:
                    name = quote["info"].get("companyName", symbol)
                    sector = quote["info"].get("industry", "Other")
            except:
                pass

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
        if self._last_fetch is None:
            return True
        return datetime.now() - self._last_fetch > self._cache_ttl
