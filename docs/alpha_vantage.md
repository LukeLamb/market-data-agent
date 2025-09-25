# Alpha Vantage API Key

Welcome to Alpha Vantage! Your dedicated access key is: 33ZRVJ15999ZL3EI. Please record this API key at a safe place for future data access.

Claude Code
To install and configure (replace YOUR_API_KEY with your actual Alpha Vantage API key):

claude mcp add -t http alphavantage <https://mcp.alphavantage.co/mcp?apikey=YOUR_API_KEY>
Run claude in your terminal from your project directory

Then connect with:

/mcp

Category Filtering
Optionally filter available tools by category using:

Query parameter: ?categories=core_stock_apis,alpha_intelligence
Available categories:

core_stock_apis - Core stock market data APIs
options_data_apis - Options data APIs
alpha_intelligence - News sentiment and intelligence APIs
fundamental_data - Company fundamentals and financial data
forex - Foreign exchange rates and data
cryptocurrencies - Digital and crypto currencies data
commodities - Commodities and precious metals data
economic_indicators - Economic indicators and market data
technical_indicators - Technical analysis indicators and calculations
ping - Health check and utility tools
If no categories are specified, all tools will be available.

Tools Reference
Category Tools
core_stock_apis TIME_SERIES_INTRADAY, TIME_SERIES_DAILY, TIME_SERIES_DAILY_ADJUSTED, TIME_SERIES_WEEKLY, TIME_SERIES_WEEKLY_ADJUSTED, TIME_SERIES_MONTHLY, TIME_SERIES_MONTHLY_ADJUSTED, GLOBAL_QUOTE, REALTIME_BULK_QUOTES, SYMBOL_SEARCH, MARKET_STATUS
options_data_apis REALTIME_OPTIONS, HISTORICAL_OPTIONS
alpha_intelligence NEWS_SENTIMENT, EARNINGS_CALL_TRANSCRIPT, TOP_GAINERS_LOSERS, INSIDER_TRANSACTIONS, ANALYTICS_FIXED_WINDOW, ANALYTICS_SLIDING_WINDOW
fundamental_data COMPANY_OVERVIEW, INCOME_STATEMENT, BALANCE_SHEET, CASH_FLOW, EARNINGS, LISTING_STATUS, EARNINGS_CALENDAR, IPO_CALENDAR
forex FX_INTRADAY, FX_DAILY, FX_WEEKLY, FX_MONTHLY
cryptocurrencies CURRENCY_EXCHANGE_RATE, DIGITAL_CURRENCY_INTRADAY, DIGITAL_CURRENCY_DAILY, DIGITAL_CURRENCY_WEEKLY, DIGITAL_CURRENCY_MONTHLY
commodities WTI, BRENT, NATURAL_GAS, COPPER, ALUMINUM, WHEAT, CORN, COTTON, SUGAR, COFFEE, ALL_COMMODITIES
economic_indicators REAL_GDP, REAL_GDP_PER_CAPITA, TREASURY_YIELD, FEDERAL_FUNDS_RATE, CPI, INFLATION, RETAIL_SALES, DURABLES, UNEMPLOYMENT, NONFARM_PAYROLL
technical_indicators SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, MAMA, VWAP, T3, MACD, MACDEXT, STOCH, STOCHF, RSI, STOCHRSI, WILLR, ADX, ADXR, APO, PPO, MOM, BOP, CCI, CMO, ROC, ROCR, AROON, AROONOSC, MFI, TRIX, ULTOSC, DX, MINUS_DI, PLUS_DI, MINUS_DM, PLUS_DM, BBANDS, MIDPOINT, MIDPRICE, SAR, TRANGE, ATR, NATR, AD, ADOSC, OBV, HT_TRENDLINE, HT_SINE, HT_TRENDMODE, HT_DCPERIOD, HT_DCPHASE, HT_PHASOR
ping PING, ADD_TWO_NUMBERS
