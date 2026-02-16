"""
StockZone.in — Indian Stock Market Technical Indicator Zone API
Main application entry point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.routers import stocks, indicators, scanner,config
from app.services.data_service import data_service

# ============================================================
# APPLICATION LIFESPAN — startup/shutdown hooks
# ============================================================




@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: preload stock data. Shutdown: cleanup."""
    print("🚀 StockZone API starting up...")
    await data_service.initialize()
    print("✅ Stock data loaded successfully")
    yield
    print("🛑 StockZone API shutting down...")


# ============================================================
# CREATE APP
# ============================================================

app = FastAPI(
    title="StockZone.in API",
    description="Indian Stock Market Technical Indicator Zone Analysis API. "
                "Get real-time zone signals (BUY/SELL/HOLD) across 10+ technical indicators "
                "for NSE-listed stocks.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow your React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# MOUNT ROUTERS
# ============================================================

app.include_router(stocks.router, prefix="/api/v1/stocks", tags=["Stocks"])
app.include_router(indicators.router, prefix="/api/v1/indicators", tags=["Indicators"])
app.include_router(scanner.router, prefix="/api/v1/scanner", tags=["Scanner"])
app.include_router(config.router, prefix="/api/v1/config", tags=["Config"])


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
async def root():
    return {
        "name": "StockZone.in API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "stocks": "/api/v1/stocks/",
            "stock_detail": "/api/v1/stocks/{symbol}",
            "stock_zones": "/api/v1/stocks/{symbol}/zones",
            "indicators": "/api/v1/indicators/{symbol}/{indicator}",
            "scanner": "/api/v1/scanner/scan",
        },
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "stocks_loaded": data_service.is_initialized}
