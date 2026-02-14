# StockZone.in — Indian Stock Market Technical Indicator Zone API

A production-ready FastAPI backend that fetches real NSE stock data via Yahoo Finance,
computes 14 technical indicators using `pandas-ta`, and provides **zone-based signals**
(BUY/SELL/HOLD) with consensus scoring.

## 🚀 Quick Start

### Option 1: Local Setup

```bash
# Clone and enter the project
cd stockzone-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Option 2: Docker

```bash
docker-compose up --build
```

### Access the API

- **API Docs (Swagger)**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

---

## 📊 API Endpoints

### Stocks

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/stocks/` | List all stocks with zone summaries |
| GET | `/api/v1/stocks/{symbol}` | Get stock price details |
| GET | `/api/v1/stocks/{symbol}/zones` | **Full zone analysis** — all indicators |

### Indicators

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/indicators/` | List all available indicators |
| GET | `/api/v1/indicators/{symbol}/{indicator}` | Get specific indicator zone |

### Scanner

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/scanner/scan` | Screen stocks by criteria |
| GET | `/api/v1/scanner/presets` | Pre-built scan strategies |

---

## 🔍 Scanner Examples

```bash
# All stocks with RSI oversold (BUY signal)
GET /api/v1/scanner/scan?indicator=rsi&signal=BUY

# Golden Cross stocks
GET /api/v1/scanner/scan?indicator=ma_cross&zone=Golden%20Cross

# Strong buy consensus (score > 6)
GET /api/v1/scanner/scan?min_score=6&sort_by=consensus_score

# Banking stocks with bullish Supertrend
GET /api/v1/scanner/scan?indicator=supertrend&signal=BUY&sector=Banking

# Dip-buying: stocks down 2%+ today with bullish indicators
GET /api/v1/scanner/scan?max_change=-2&min_score=2
```

---

## 📈 Technical Indicators (14 Total)

| # | Indicator | Category | Zones |
|---|-----------|----------|-------|
| 1 | **RSI** | Momentum | Overbought / Bullish / Neutral / Bearish / Oversold |
| 2 | **Bollinger Bands** | Volatility | Upper Band / Above Middle / Below Middle / Lower Band |
| 3 | **MACD** | Trend | Strong Bullish / Bullish Cross / Bearish Cross / Strong Bearish |
| 4 | **Pivot Points** | Support/Resist | Above R2 / R1-R2 / Above Pivot / Below Pivot / S1-S2 / Below S2 |
| 5 | **Supertrend** | Trend | Bullish / Bearish |
| 6 | **Stochastic** | Momentum | Overbought / Bullish / Bearish / Oversold |
| 7 | **ADX** | Trend | Strong Trend / Trending / Weak/No Trend |
| 8 | **CCI** | Momentum | Extreme Overbought / Overbought / Neutral / Oversold / Extreme Oversold |
| 9 | **VWAP** | Volume | Above / Near Above / Near Below / Below |
| 10 | **MA Cross (50/200)** | Trend | Golden Cross / Death Cross / Above SMA50 / Below SMA50 |
| 11 | **Williams %R** | Momentum | Overbought / Upper Half / Lower Half / Oversold |
| 12 | **ATR** | Volatility | High / Moderate / Low Volatility |
| 13 | **EMA Cross (12/26)** | Trend | Bullish Cross / EMA12 Above / EMA12 Below / Bearish Cross |
| 14 | **OBV** | Volume | Accumulation / Distribution / Rising / Falling |

---

## 🏗️ Architecture

```
stockzone-backend/
├── main.py                          # FastAPI app + lifespan
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── app/
│   ├── config.py                    # NIFTY 50 stock universe
│   ├── models/
│   │   └── __init__.py              # Pydantic schemas
│   ├── routers/
│   │   ├── stocks.py                # /stocks endpoints
│   │   ├── indicators.py            # /indicators endpoints
│   │   └── scanner.py               # /scanner endpoints
│   └── services/
│       ├── data_service.py          # Yahoo Finance data fetcher + cache
│       └── indicator_service.py     # Technical indicator engine (pandas-ta)
```

---

## 🔧 Configuration

### Adding More Stocks

Edit `app/config.py` to add/remove stocks:

```python
NIFTY50_STOCKS = {
    "SYMBOL": {
        "name": "Full Company Name",
        "sector": "Sector",
        "yahoo": "SYMBOL.NS",  # .NS for NSE, .BO for BSE
    },
}
```

### Cache TTL

In `data_service.py`, adjust refresh interval:

```python
self._cache_ttl = timedelta(minutes=5)  # Default: 5 minutes
```

---

## 🔄 Connecting to the React Frontend

The React dashboard (stock-zone-dashboard.jsx) needs to be updated to fetch from this API
instead of using mock data. Here's how:

```javascript
// Replace the mock data in the React app with API calls:

// Fetch all stocks for the table
const response = await fetch('http://localhost:8000/api/v1/stocks/?sort_by=consensus_score&sort_order=desc');
const data = await response.json();
// data.stocks → array of StockListItem

// Fetch full zone analysis when user clicks a stock
const zonesResponse = await fetch(`http://localhost:8000/api/v1/stocks/${symbol}/zones`);
const zones = await zonesResponse.json();
// zones.zones → array of IndicatorZone with zone details

// Run a scan
const scanResponse = await fetch('http://localhost:8000/api/v1/scanner/scan?indicator=rsi&signal=BUY');
const scanResults = await scanResponse.json();
```

---

## 🚀 Production Deployment Tips

1. **Use Redis** for caching — uncomment in docker-compose.yml
2. **Use APScheduler** for periodic data refresh (every 1 min during market hours)
3. **Rate Limiting** — yfinance has limits; consider paid APIs for production:
   - **Kite Connect** (Zerodha) — real-time streaming
   - **Angel One SmartAPI** — free tier available
   - **Upstox API** — WebSocket support
4. **Database** — Add PostgreSQL/TimescaleDB to store historical indicator data
5. **WebSocket** — Add for live price/zone updates to the frontend
6. **Deploy** on AWS/GCP/Railway/Render with Gunicorn + Uvicorn workers

---

## ⚠️ Disclaimer

This is for educational and informational purposes only. Not financial advice.
Always do your own research and consult a financial advisor before investing.
Past indicator performance does not guarantee future results.
