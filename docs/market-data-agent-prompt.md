# Market Data Agent - Development Prompt

## Project Overview

You are tasked with building a sophisticated Market Data Agent that serves as the foundational data layer for a comprehensive trading system. This agent must provide clean, reliable, and real-time market data to other trading agents while maintaining high availability and data quality standards.

## Core Objectives

### Primary Goals

- Establish robust data pipeline from multiple financial data sources
- Ensure data quality, consistency, and reliability across all sources
- Provide both real-time streaming and historical data access
- Implement intelligent failover and redundancy mechanisms
- Create efficient data storage and retrieval systems

### Success Metrics

- 99.9% uptime for data collection
- Sub-100ms response time for cached data queries
- Zero data gaps during market hours
- Automatic recovery from API failures within 30 seconds
- Cost optimization through intelligent API usage

## Technical Architecture

### Core Components

#### 1. Data Sources Manager

**Responsibility**: Manage connections to multiple financial data APIs
**Key Features**:

- Multi-source configuration management
- Authentication and credential rotation
- Connection pooling and persistent connections
- Health monitoring of each data source
- Automatic source prioritization based on reliability

**Implementation Details**:

```python
class DataSourceManager:
    def __init__(self):
        self.sources = {}  # {source_name: source_config}
        self.health_scores = {}  # {source_name: reliability_score}
        
    def add_source(self, name, config):
        # Add new data source with configuration
        
    def get_best_source(self, data_type):
        # Return highest reliability source for specific data type
        
    def failover_sequence(self, data_type):
        # Return ordered list of fallback sources
```

#### 2. Data Validator

**Responsibility**: Ensure data quality and consistency
**Validation Rules**:

- Price data: No negative prices, reasonable bid-ask spreads
- Volume data: Non-negative, within historical ranges
- Timestamp validation: Proper market hours, no future dates
- Cross-source validation: Compare similar data points
- Statistical outlier detection: Flag unusual price movements

**Quality Scoring System**:

- A+ (95-100%): Perfect data, no issues detected
- A (90-94%): Minor inconsistencies, acceptable quality
- B (80-89%): Some issues, requires monitoring
- C (70-79%): Multiple issues, use with caution
- F (<70%): Poor quality, reject or flag for manual review

#### 3. Storage Handler

**Responsibility**: Efficient data storage and retrieval
**Storage Strategy**:

- **Time-series database** for OHLCV data (InfluxDB or TimescaleDB)
- **Relational database** for metadata and configurations (PostgreSQL)
- **In-memory cache** for real-time data (Redis)
- **File storage** for bulk historical downloads

**Data Partitioning**:

- Partition by symbol and date for optimal query performance
- Separate tables for different data types (stocks, forex, crypto)
- Implement data retention policies (e.g., tick data for 30 days, daily data indefinitely)

#### 4. Rate Limiter

**Responsibility**: Manage API quotas and request timing
**Features**:

- Per-source rate limiting with different tiers
- Intelligent batching of requests
- Request scheduling based on priority
- Quota usage monitoring and alerts
- Automatic throttling when approaching limits

#### 5. Real-time Streamer

**Responsibility**: Handle live market data feeds
**Capabilities**:

- WebSocket connections for real-time feeds
- Message parsing and normalization
- Real-time data validation
- Immediate distribution to subscribers
- Connection recovery and reconnection logic

#### 6. Historical Fetcher

**Responsibility**: Bulk download and backfill historical data
**Features**:

- Chunked downloads to respect API limits
- Gap detection and automatic backfilling
- Progress tracking for large downloads
- Resume capability for interrupted downloads
- Data integrity verification post-download

## MCP Server Integration

### Memory Server Usage

Store and track the following entities and relationships:

**Entities**:

```bash
- data_source (name: "yfinance", reliability: 0.95, cost_per_call: 0.0)
- symbol (ticker: "AAPL", exchange: "NASDAQ", sector: "Technology")
- market_regime (type: "bull_market", start_date: "2023-01-01", confidence: 0.8)
- data_quality_event (timestamp, source, issue_type, severity)
```

**Relationships**:

```bash
- data_source -> provides -> symbol_data
- symbol -> belongs_to -> sector
- market_regime -> affects -> symbol_volatility
- data_quality_event -> impacts -> data_source_reliability
```

### Sequential Thinking Integration

Use for complex data processing workflows:

**Data Validation Workflow**:

1. Fetch data from primary source
2. Perform basic validation checks
3. Calculate quality score
4. If score < threshold, fetch from secondary source
5. Compare data between sources
6. Store final data with quality metadata
7. Update source reliability scores

### Time Server Usage

- Market hours validation
- Timezone conversions for global markets
- Trading calendar management
- Data timestamp normalization

## Data Sources Configuration

### Tier 1 Sources (High Reliability, Paid)

- **Polygon.io**: Real-time and historical stock/crypto data
- **IEX Cloud**: Reliable stock data with good free tier
- **Alpha Vantage**: Comprehensive data including forex and commodities

### Tier 2 Sources (Good Reliability, Free/Cheap)

- **yfinance**: Yahoo Finance data, excellent for backtesting
- **FRED**: Federal Reserve economic data
- **Financial Modeling Prep**: Fundamental data

### Tier 3 Sources (Backup/Specialized)

- **Quandl**: Alternative datasets
- **NewsAPI**: Sentiment data
- **Yahoo Finance RSS**: News feeds

## Implementation Phases

### Phase 1: Foundation

- [ ] Set up basic data source connections (yfinance, Alpha Vantage)
- [ ] Implement simple SQLite storage
- [ ] Create basic OHLCV data fetching
- [ ] Add simple data validation
- [ ] Test with 5-10 popular stocks

### Phase 2: Reliability

- [ ] Add multiple data sources with failover
- [ ] Implement rate limiting
- [ ] Add comprehensive data validation
- [ ] Create data quality scoring system
- [ ] Add memory server integration for tracking

### Phase 3: Performance

- [ ] Implement time-series database (InfluxDB or TimescaleDB)
- [ ] Add Redis caching layer
- [ ] Optimize query performance
- [ ] Add real-time streaming capabilities
- [ ] Implement bulk historical data loading

### Phase 4: Production Ready

- [ ] Add comprehensive error handling and logging
- [ ] Implement monitoring and alerting
- [ ] Add configuration management
- [ ] Create API endpoints for other agents
- [ ] Add comprehensive testing suite

## Data Schema Design

### OHLCV Table Structure

```sql
CREATE TABLE ohlcv_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(10,2) NOT NULL,
    high DECIMAL(10,2) NOT NULL,
    low DECIMAL(10,2) NOT NULL,
    close DECIMAL(10,2) NOT NULL,
    volume BIGINT NOT NULL,
    adjusted_close DECIMAL(10,2),
    source VARCHAR(50) NOT NULL,
    quality_score INTEGER CHECK (quality_score >= 0 AND quality_score <= 100),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, timestamp, source)
);
```

### Data Quality Events

```sql
CREATE TABLE data_quality_events (
    id BIGSERIAL PRIMARY KEY,
    source VARCHAR(50) NOT NULL,
    symbol VARCHAR(10),
    timestamp TIMESTAMPTZ NOT NULL,
    event_type VARCHAR(50) NOT NULL, -- 'missing_data', 'outlier', 'api_error'
    severity VARCHAR(20) NOT NULL,   -- 'low', 'medium', 'high', 'critical'
    description TEXT,
    resolved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## API Design for Other Agents

### REST Endpoints

```python
# Get current price
GET /api/v1/price/{symbol}

# Get historical data
GET /api/v1/historical/{symbol}?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD

# Get multiple symbols
POST /api/v1/batch/prices
Body: {"symbols": ["AAPL", "GOOGL", "MSFT"]}

# Health check
GET /api/v1/health

# Data quality report
GET /api/v1/quality/{symbol}?days=30
```

### WebSocket Endpoints

```python
# Real-time price feed
WS /api/v1/stream/prices
Subscribe: {"action": "subscribe", "symbols": ["AAPL", "GOOGL"]}
```

## Error Handling Strategy

### API Failures

- Implement exponential backoff for temporary failures
- Switch to backup sources for extended outages
- Log all failures with context for analysis
- Notify other agents of data availability issues

### Data Quality Issues

- Flag suspicious data but don't automatically reject
- Provide confidence scores with all data
- Allow manual override for quality thresholds
- Maintain audit trail of all quality decisions

## Testing Strategy

### Unit Tests

- Test each component in isolation
- Mock external API calls
- Validate data transformation logic
- Test error handling scenarios

### Integration Tests

- Test full data pipeline with real APIs (using test accounts)
- Validate data quality across different sources
- Test failover mechanisms
- Performance testing with high-volume requests

### Load Testing

- Test system under peak market conditions
- Validate rate limiting effectiveness
- Test memory usage under continuous operation
- Stress test database performance

## Configuration Management

### Environment Variables

```bash
# API Keys
ALPHA_VANTAGE_API_KEY=your_key_here
POLYGON_API_KEY=your_key_here
IEX_CLOUD_TOKEN=your_token_here

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/trading_data
REDIS_URL=redis://localhost:6379

# Rate Limits
ALPHA_VANTAGE_CALLS_PER_MINUTE=5
POLYGON_CALLS_PER_MINUTE=1000
```

### Configuration Files

```yaml
# config.yaml
data_sources:
  yfinance:
    enabled: true
    priority: 2
    cost_per_call: 0.0
    rate_limit: 2000  # per hour
    
  alpha_vantage:
    enabled: true
    priority: 1
    cost_per_call: 0.0
    rate_limit: 500   # per day
```

## Monitoring and Alerting

### Key Metrics to Track

- API response times by source
- Data quality scores over time
- API quota usage and remaining limits
- Number of failover events
- Database query performance
- Memory and CPU usage

### Alert Conditions

- Any API source down for >5 minutes
- Data quality score drops below 80% for any symbol
- API quota usage exceeds 90%
- Database connection failures
- Memory usage exceeds 80%

## Security Considerations

### API Key Management

- Store API keys in environment variables or secure vault
- Implement key rotation capabilities
- Monitor for key usage anomalies
- Use least-privilege principle for database access

### Data Privacy

- No storage of personally identifiable information
- Secure transmission of all data
- Regular security audits of dependencies
- Implement proper access controls

## Development Guidelines

### Code Quality Standards

- Follow PEP 8 for Python code style
- Minimum 90% test coverage
- Type hints for all function parameters and returns
- Comprehensive docstrings for all public methods
- Regular code reviews before merging

### Documentation Requirements

- API documentation with examples
- Architecture decision records (ADRs)
- Deployment and configuration guides
- Troubleshooting guides
- Performance tuning recommendations

## Success Criteria

### Technical Metrics

- [ ] 99.9% uptime during market hours
- [ ] Sub-100ms response time for cached queries
- [ ] Zero data gaps longer than 1 minute
- [ ] Automatic recovery from failures in <30 seconds
- [ ] Data quality score >90% across all sources

### Business Metrics

- [ ] API costs under $100/month for basic setup
- [ ] Support for 100+ symbols simultaneously
- [ ] Historical data coverage for 5+ years
- [ ] Real-time data latency <500ms
- [ ] Successful integration with 3+ downstream agents

## Next Steps After Completion

Once the Market Data Agent is complete and tested:

1. **Integration Testing**: Test with a simple pattern recognition agent
2. **Performance Optimization**: Fine-tune based on real usage patterns
3. **Feature Extensions**: Add support for options data, crypto, forex
4. **Documentation**: Create comprehensive user guides
5. **Monitoring Setup**: Implement production monitoring dashboard

## Resources and References

### Technical Documentation

- [Pandas Time Series](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [InfluxDB Python Client](https://influxdb-client.readthedocs.io/en/stable/)
- [Redis Python Guide](https://redis-py.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

### Financial Data APIs

- [Alpha Vantage API Docs](https://www.alphavantage.co/documentation/)
- [Polygon.io API Docs](https://polygon.io/docs/)
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [IEX Cloud API](https://iexcloud.io/docs/api/)

### Best Practices

- [Financial Data Standards](https://www.fpml.org/)
- [Time Series Database Design](https://docs.influxdata.com/influxdb/v2.0/reference/syntax/)
- [Rate Limiting Strategies](https://cloud.google.com/architecture/rate-limiting-strategies-techniques)

---

**Remember**: This is a foundational component. Focus on reliability and data quality over features. Other agents will depend on this data, so accuracy and availability are paramount.
