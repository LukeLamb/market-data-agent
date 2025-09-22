# Market Data Agent

A sophisticated market data collection and validation system that serves as the foundational data layer for trading applications.

## Features

- **Multi-source data collection** from Yahoo Finance, Alpha Vantage, and other providers
- **Intelligent failover** and redundancy mechanisms
- **Data quality validation** with scoring system
- **Real-time and historical data** access
- **RESTful API** for easy integration
- **SQLite storage** with efficient querying

## Quick Start

### Prerequisites

- Python 3.8 or higher
- API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key) (free)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/LukeLamb/market-data-agent.git
cd market-data-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. Run the application:
```bash
python main.py
```

## API Endpoints

- `GET /health` - System health check
- `GET /price/{symbol}` - Current price for a symbol
- `GET /historical/{symbol}` - Historical data with date range
- `GET /sources/status` - Data source health status

## Architecture

The system is built with a modular architecture:

- **Data Sources**: Pluggable connectors for different APIs
- **Storage**: SQLite database with efficient indexing
- **Validation**: Comprehensive data quality checks
- **API**: FastAPI-based REST interface

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black src/ tests/
flake8 src/ tests/
```

### Type Checking

```bash
mypy src/
```

## License

MIT License - see LICENSE file for details.