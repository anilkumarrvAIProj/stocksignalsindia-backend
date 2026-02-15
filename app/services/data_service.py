"""
DataService — fetches stock data from Fyers API.
Reliable, works on cloud servers (Render, Railway, etc.)
Requires FYERS_CLIENT_ID and FYERS_ACCESS_TOKEN environment variables.
"""

import pandas as pd
import numpy as np
from fyers_apiv3 import fyersModel
from datetime import datetime, timedelta
from typing import Optional
import asyncio
import logging
import time
import random
import os

from app.config import NIFTY50_STOCKS

logger = logging.getLogger(__name__)


class DataService:
    def __init__(self):
        self.is_initialized = False
        self._cache: dict[str, dict] = {}
        self._ohlcv_cache: dict[str, pd.DataFrame] = {}
        self._last_fetch: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=5)
        self._custom_symbols: set = set()
        self._fyers = None
        self._init_fyers()

    def _init_fyers(self):
        """Initialize Fyers API client from environment variables."""
        client_id = os.environ.get("FYERS_CLIENT_ID", "AZF2T7MBGP-100")
        access_token = os.environ.get("FYERS_ACCESS_TOKEN", "")

        if not access_token:
            logger.error("❌ FYERS_ACCESS_TOKEN not set! Add it in Render Environment variables.")
            return

        try:
            self._fyers = fyersModel.FyersModel(
                client_id=client_id,
                token=access_token,
                is_async=False,
                log_path=""
            )
            # Test connection
            profile = self._fyers.get_profile()
            if profile.get("s") == "ok":
                logger.info(f"✅ Fyers connected: {profile.get('data', {}).get('name', 'Unknown')}")
            else:
                logger.warning(f"⚠ Fyers auth issue: {profile}")
        except Exception as e:
            logger.error(f"❌ Fyers init failed: {e}")
            self._fyers = None

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
        if not self._fyers:
            logger.error("❌ Fyers not initialized. Cannot fetch data.")
            return

        total = len(NIFTY50_STOCKS)
        logger.info(f"📊 Fetching {total} stocks from Fyers API...")

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

        success_count = 0
        failed_symbols = []

        for i, (nse_symbol, info) in enumerate(NIFTY50_STOCKS.items()):
            try:
                # Small delay to respect rate limits
                if i > 0 and i % 10 == 0:
                    time.sleep(1)

                # Fyers symbol format: NSE:SYMBOL-EQ
                fyers_symbol = f"NSE:{nse_symbol}-EQ"

                data = {
                    "symbol": fyers_symbol,
                    "resolution": "D",
                    "date_format": "1",
                    "range_from": start_date,
                    "range_to": end_date,
                    "cont_flag": "1"
                }

                response = self._fyers.history(data=data)

                if response.get("s") != "ok" or "candles" not in response:
                    failed_symbols.append(nse_symbol)
                    logger.debug(f"[{i+1}/{total}] ❌ {nse_symbol}: {response.get('message', 'No data')}")
                    continue

                candles = response["candles"]
                if not candles:
                    failed_symbols.append(nse_symbol)
                    continue

                # candles format: [[timestamp, open, high, low, close, volume], ...]
                df = pd.DataFrame(candles, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
                df["Date"] = pd.to_datetime(df["Date"], unit="s")
                df = df.sort_values("Date").reset_index(drop=True)

                self._ohlcv_cache[nse_symbol] = df.copy()
                self._process_stock_data(nse_symbol, df, info["name"], info["sector"])
                success_count += 1

                if success_count % 10 == 0 or success_count == 1:
                    logger.info(f"[{i+1}/{total}] ✅ {nse_symbol}: ₹{self._cache[nse_symbol]['price']} ({success_count} loaded)")

            except Exception as e:
                failed_symbols.append(nse_symbol)
                logger.warning(f"[{i+1}/{total}] ❌ {nse_symbol}: {str(e)[:80]}")

        # Retry failed ones
        if failed_symbols:
            logger.info(f"🔄 Retrying {len(failed_symbols)} failed stocks...")
            time.sleep(2)
            for nse_symbol in failed_symbols:
                info = NIFTY50_STOCKS.get(nse_symbol)
                if not info:
                    continue
                try:
                    fyers_symbol = f"NSE:{nse_symbol}-EQ"
                    data = {
                        "symbol": fyers_symbol,
                        "resolution": "D",
                        "date_format": "1",
                        "range_from": start_date,
                        "range_to": end_date,
                        "cont_flag": "1"
                    }
                    response = self._fyers.history(data=data)
                    if response.get("s") == "ok" and response.get("candles"):
                        df = pd.DataFrame(response["candles"], columns=["Date", "Open", "High", "Low", "Close", "Volume"])
                        df["Date"] = pd.to_datetime(df["Date"], unit="s")
                        df = df.sort_values("Date").reset_index(drop=True)
                        self._ohlcv_cache[nse_symbol] = df.copy()
                        self._process_stock_data(nse_symbol, df, info["name"], info["sector"])
                        success_count += 1
                        logger.info(f"[Retry] ✅ {nse_symbol}")
                except Exception:
                    pass

        logger.info(f"✅ TOTAL: {len(self._cache)}/{total} stocks loaded via Fyers API")

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
        }

    # ============================================================
    # ON-DEMAND FETCHING
    # ============================================================

    def fetch_on_demand_sync(self, symbol: str) -> Optional[dict]:
        symbol = symbol.upper().strip()
        if symbol in self._cache:
            return self._cache[symbol]

        if not self._fyers:
            return None

        logger.info(f"On-demand fetch: {symbol}")
        try:
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
                return None

            df = pd.DataFrame(response["candles"], columns=["Date", "Open", "High", "Low", "Close", "Volume"])
            df["Date"] = pd.to_datetime(df["Date"], unit="s")
            df = df.sort_values("Date").reset_index(drop=True)

            # Try to get company name from quotes
            name = symbol
            sector = "Other"
            try:
                quote_data = {"symbols": fyers_symbol}
                quote_resp = self._fyers.quotes(quote_data)
                if quote_resp.get("s") == "ok" and quote_resp.get("d"):
                    q = quote_resp["d"][0]
                    name = q.get("n", symbol)
                    # Map sector from short_name
                    sector_map = {
                        "Financial Services": "Banking",
                        "Information Technology": "IT",
                        "Consumer Defensive": "FMCG",
                        "Consumer Cyclical": "Consumer",
                        "Basic Materials": "Metals",
                        "Communication Services": "Telecom",
                    }
                    raw_sector = q.get("v", {}).get("sector", "Other")
                    sector = sector_map.get(raw_sector, raw_sector) if raw_sector else "Other"
            except Exception:
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
