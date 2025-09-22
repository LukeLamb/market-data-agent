# Phase 1: Foundation Implementation Guide

## Overview

This phase establishes the core foundation of the Market Data Agent with minimal complexity. The focus is on getting a working system that can fetch, validate, and store basic market data before adding advanced features.

## Success Criteria

- Reliable data fetching from 2+ sources
- Basic data validation and quality scoring
- Simple but effective storage system
- Support for 5-10 popular stocks
- Functional API endpoints for testing

## Implementation Steps

### Step 1: Project Structure Setup

Create the basic project structure:

```bash
market_data_agent/
├── src/
│   ├── __init__.py
│   ├── data_sources/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── yfinance_source.py
│   │   └── alpha_vantage_source.py
│   ├── storage/
│   │   ├── __init__.py
│   │   └── sqlite_handler.py
│   ├── validation/
│   │   ├── __init__.py
│   │   └── data_validator.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── endpoints.py
│   └── config/
│       ├── __init__.py
│       └── settings.py
├── tests/
├── requirements.txt
└── main.py
```

### Step 2: Environment and Dependencies Setup

**Requirements to install:**

- `yfinance` - Yahoo Finance data
- `requests` - HTTP requests for Alpha Vantage
- `sqlite3` - Built-in SQLite database
- `fastapi` - API framework
- `uvicorn` - ASGI server
- `pandas` - Data manipulation
- `python-dotenv` - Environment variable management
- `pydantic` - Data validation

**Environment variables needed:**

- `ALPHA_VANTAGE_API_KEY` (get free key from alphavantage.co)

### Step 3: Base Data Source Interface

Create abstract base class for all data sources to ensure consistency:

**Key methods to implement:**

- `get_current_price(symbol)` - Get real-time price
- `get_historical_data(symbol, start_date, end_date)` - Get OHLCV data
- `get_health_status()` - Check if source is working
- `get_supported_symbols()` - List of available symbols

### Step 4: YFinance Data Source

Implement Yahoo Finance connector:

**Features:**

- Free, no API key required
- Good historical data coverage
- Real-time delayed prices (15-20 minutes)
- Support for stocks, ETFs, indices

**Limitations to handle:**

- Rate limiting (unofficial limits)
- Potential reliability issues
- No guaranteed SLA

### Step 5: Alpha Vantage Data Source

Implement Alpha Vantage connector:

**Features:**

- Free tier: 25 requests/day
- Real-time and historical data
- Good reliability
- Official API with documentation

**Configuration needed:**

- API key management
- Request rate limiting (5 calls/minute)
- Error handling for quota exceeded

### Step 6: Simple SQLite Storage

Create basic storage system:

**Tables to create:**

1. `ohlcv_data` - Price and volume data
2. `data_sources` - Source configuration and health
3. `symbols` - Symbol metadata
4. `quality_events` - Data quality issues

**Key features:**

- Efficient queries by symbol and date range
- Data deduplication handling
- Basic indexing for performance

### Step 7: Data Validation Framework

Implement basic validation rules:

**Price validation:**

- No negative prices
- No zero prices during market hours
- Price changes within reasonable bounds (e.g., <50% in one period)

**Volume validation:**

- Non-negative volume
- Volume within historical ranges

**Timestamp validation:**

- No future timestamps
- Proper market hours consideration

**Quality scoring (simplified):**

- Start with binary: GOOD (90-100%) or BAD (0-89%)
- Later expand to full A-F system

### Step 8: Data Source Manager

Create simple source management:

**Features:**

- Primary/backup source configuration
- Automatic failover on errors
- Basic health monitoring
- Source reliability tracking

**Simple failover logic:**

1. Try primary source (yfinance)
2. If fails, try secondary (Alpha Vantage)
3. Log all failures for analysis
4. Update source reliability scores

### Step 9: Basic API Endpoints

Create minimal REST API:

**Endpoints to implement:**

- `GET /health` - System health check
- `GET /price/{symbol}` - Current price
- `GET /historical/{symbol}` - Historical data with date range
- `GET /sources/status` - Data source health status

**Response format:**

- JSON with data + metadata (source, quality_score, timestamp)
- Consistent error handling
- Proper HTTP status codes

### Step 10: Configuration Management

Simple configuration system:

**Config file structure:**

```yaml
sources:
  yfinance:
    enabled: true
    priority: 1
    timeout: 30
  alpha_vantage:
    enabled: true
    priority: 2
    timeout: 30
    rate_limit: 5  # per minute

storage:
  database_path: "data/market_data.db"

validation:
  max_price_change_percent: 50
  min_volume: 0
```

### Step 11: Basic Error Handling

Implement essential error handling:

**Error types to handle:**

- Network timeouts
- API rate limit exceeded
- Invalid symbol requests
- Database connection issues
- Data validation failures

**Error responses:**

- Log all errors with context
- Return user-friendly error messages
- Implement retry logic for transient failures

### Step 12: Simple Testing Setup

Create basic test coverage:

**Test categories:**

- Data source connectivity tests
- Data validation tests
- Database storage/retrieval tests
- API endpoint tests

**Mock external APIs:**

- Use test data for reliable testing
- Test error scenarios
- Validate data transformations

## Testing Strategy for Phase 1

### Manual Testing Checklist

**Data Sources:**

- [ ] YFinance can fetch AAPL current price
- [ ] YFinance can fetch AAPL historical data (1 month)
- [ ] Alpha Vantage can fetch AAPL current price (with API key)
- [ ] Failover works when primary source fails

**Storage:**

- [ ] Data saves to SQLite correctly
- [ ] Can retrieve saved data by symbol and date
- [ ] Duplicate data handling works
- [ ] Database queries are reasonably fast (<100ms)

**Validation:**

- [ ] Rejects negative prices
- [ ] Flags unusual price changes
- [ ] Validates timestamp formats
- [ ] Assigns quality scores correctly

**API:**

- [ ] All endpoints return valid JSON
- [ ] Error handling works for invalid symbols
- [ ] Health endpoint shows system status
- [ ] Response times are acceptable

### Test Symbols for Validation

Use these symbols for testing (liquid, well-known stocks):

- AAPL (Apple)
- GOOGL (Google)
- MSFT (Microsoft)
- TSLA (Tesla)
- SPY (S&P 500 ETF)

## Success Metrics for Phase 1

**Functional Requirements:**

- [ ] Successfully fetch data from both sources
- [ ] Store and retrieve data for all test symbols
- [ ] API responds to all endpoints
- [ ] Basic validation catches obvious bad data
- [ ] Failover mechanism works

**Performance Requirements:**

- [ ] API response time <500ms for cached data
- [ ] Database queries complete in <100ms
- [ ] Can handle 10 concurrent requests
- [ ] Memory usage stays reasonable during operation

**Quality Requirements:**

- [ ] No crashes during normal operation
- [ ] Proper error messages for all failure cases
- [ ] Data quality score >90% for good sources
- [ ] Zero data corruption in storage

## Known Limitations in Phase 1

**Intentional Simplifications:**

- SQLite instead of time-series database
- Simple failover instead of intelligent routing
- Binary quality scoring instead of full A-F system
- No real-time streaming (polling only)
- Limited error recovery mechanisms
- No rate limiting implementation
- Basic configuration (no hot reloading)

**These will be addressed in later phases.**

## Next Steps After Phase 1

Once Phase 1 is complete and tested:

1. **Code Review** - Review all code for quality and consistency
2. **Performance Testing** - Test with larger datasets and more symbols
3. **Documentation** - Document all APIs and configuration options
4. **Phase 2 Planning** - Plan reliability and scaling improvements

## Dependencies and Prerequisites

**Before starting Phase 1:**

- [ ] Python 3.8+ installed
- [ ] Git repository initialized
- [ ] Alpha Vantage API key obtained (free)
- [ ] Development environment set up
- [ ] Required dependencies identified

**Estimated effort:** 3-5 days for experienced developer

---

**Important:** Focus on getting each step working before moving to the next. It's better to have a simple, working system than a complex, broken one.
