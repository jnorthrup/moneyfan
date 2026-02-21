# Coinbase Backfill Agent

## Overview

The Coinbase Backfill Agent (`coinbase_backfill_agent.py`) is a specialized tool for backfeeding Coinbase candle data into DuckDB with proper provenance tracking. It follows the same patterns as the existing `provenance_import.py` but focuses exclusively on Coinbase data.

## Features

1. **Arrow File Discovery**: Automatically finds Coinbase-style files in `hrm/data/arrow`
2. **DuckDB Integration**: Checks existing data and identifies gaps
3. **48-Column Schema**: Transforms data to the full 48-column schema required for live HRM agents
4. **Provenance Tracking**: Tracks source, import timestamp, data hash, and configuration
5. **Multi-Timeframe Support**: Handles 1m, 5m, 15m, and 1h timeframes
6. **Error Handling**: Graceful error handling with detailed reporting

## Installation

The agent requires:
- `hrm.duck_store.DuckStore` for DuckDB operations
- `pandas` for data manipulation
- `numpy` for calculations
- `duckdb` for database operations

## Usage

### Basic Usage

```python
from coinbase_backfill_agent import CoinbaseBackfillAgent, CoinbaseBackfillConfig

# Create configuration
config = CoinbaseBackfillConfig(
    duck_db_path="hrm/data/market.duckdb",
    coinbase_pairs=["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD"],
    timeframes=["1m", "5m", "15m", "1h"]
)

# Initialize agent
agent = CoinbaseBackfillAgent(config)

# Check arrow files
coinbase_files = agent.check_arrow_files()

# Check DuckDB gaps
gaps = agent.check_duckdb_gaps()

# Backfill data
backfill_results = agent.backfill_from_arrow(coinbase_files)

# Generate report
report = agent.generate_report()
print(report)
```

### Command Line Usage

```bash
python coinbase_backfill_agent.py
```

## Configuration

### CoinbaseBackfillConfig

- **arrow_sources**: List of directories to search for arrow files
- **duck_db_path**: Path to DuckDB database
- **coinbase_pairs**: List of Coinbase symbols to backfill (e.g., "BTC-USD", "ETH-USD")
- **timeframes**: List of timeframes to process (e.g., ["1m", "5m", "15m", "1h"])
- **schema_columns**: List of 48 columns for the schema

## 48-Column Schema

The agent creates data in the following schema:

### Basic OHLCV (5 columns)
1. `open`
2. `high`
3. `low`
4. `close`
5. `volume`

### Binance-Specific (4 columns)
6. `quote_volume`
7. `trades`
8. `taker_buy_base`
9. `taker_buy_quote`

### Technical Indicators (15 columns)
10-24. `sma_5`, `sma_15`, `sma_60`, `ema_5`, `ema_15`, `ema_60`, `rsi_14`, `macd`, `macd_signal`, `macd_hist`, `bb_upper`, `bb_lower`, `bb_mid`, `atr_14`, `adx_14`

### Synthetic Orderbook (10 columns)
25-34. `ob_imbalance`, `bid_price`, `ask_price`, `bid_size`, `ask_size`, `depth_5_bid`, `depth_5_ask`, `mid_price`, `spread_pct`, `vwap`

### Returns (4 columns)
35-38. `returns_1m`, `returns_5m`, `returns_15m`, `returns_1h`

### Volatility (1 column)
39. `vol_5m`

### Regime & Labels (3 columns)
40-42. `regime_label`, `stochastic_compass`, `horizon_tag`

### Predictor Confidences (3 columns)
43-45. `predictor_conf_5m`, `predictor_conf_15m`, `predictor_conf_1h`

### HRM-Specific (4 columns)
46-49. `hrm_reward`, `veto_flag`, `position_size_usd`, `equity_curve`

## Database Schema

### coinbase_source Table

```sql
CREATE TABLE coinbase_source (
    symbol TEXT,
    timestamp TIMESTAMP,
    timeframe TEXT,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    quote_volume DOUBLE,
    trades DOUBLE,
    taker_buy_base DOUBLE,
    taker_buy_quote DOUBLE,
    sma_5 DOUBLE,
    sma_15 DOUBLE,
    sma_60 DOUBLE,
    ema_5 DOUBLE,
    ema_15 DOUBLE,
    ema_60 DOUBLE,
    rsi_14 DOUBLE,
    macd DOUBLE,
    macd_signal DOUBLE,
    macd_hist DOUBLE,
    bb_upper DOUBLE,
    bb_lower DOUBLE,
    bb_mid DOUBLE,
    atr_14 DOUBLE,
    adx_14 DOUBLE,
    ob_imbalance DOUBLE,
    bid_price DOUBLE,
    ask_price DOUBLE,
    bid_size DOUBLE,
    ask_size DOUBLE,
    depth_5_bid DOUBLE,
    depth_5_ask DOUBLE,
    mid_price DOUBLE,
    spread_pct DOUBLE,
    vwap DOUBLE,
    returns_1m DOUBLE,
    returns_5m DOUBLE,
    returns_15m DOUBLE,
    returns_1h DOUBLE,
    vol_5m DOUBLE,
    regime_label DOUBLE,
    stochastic_compass DOUBLE,
    horizon_tag TEXT,
    predictor_conf_5m DOUBLE,
    predictor_conf_15m DOUBLE,
    predictor_conf_1h DOUBLE,
    hrm_reward DOUBLE,
    veto_flag BOOLEAN,
    position_size_usd DOUBLE,
    equity_curve DOUBLE,
    source_file TEXT,
    import_timestamp TIMESTAMP,
    data_hash TEXT,
    PRIMARY KEY (symbol, timestamp, timeframe)
)
```

### provenance_metadata Table

```sql
CREATE TABLE provenance_metadata (
    id INTEGER PRIMARY KEY,
    source_exchange TEXT,
    source_file TEXT,
    import_timestamp TIMESTAMP,
    data_timestamp_start TIMESTAMP,
    data_timestamp_end TIMESTAMP,
    row_count INTEGER,
    data_hash TEXT,
    config JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

### market_data View

```sql
CREATE VIEW market_data AS
SELECT 
    symbol,
    timestamp,
    open,
    high,
    low,
    close,
    volume,
    'coinbase' as source_exchange,
    source_file,
    import_timestamp
FROM coinbase_source
```

## Provenance Tracking

Each backfill operation is tracked with:

- **source_exchange**: "coinbase"
- **source_file**: Path to the feather file
- **import_timestamp**: When the data was imported
- **data_timestamp_start**: First timestamp in the dataset
- **data_timestamp_end**: Last timestamp in the dataset
- **row_count**: Number of rows imported
- **data_hash**: SHA256 hash of the data
- **config**: JSON serialization of the configuration

## Testing

### Unit Tests

Run the test suite:

```bash
python test_coinbase_backfill.py
```

### Specific File Test

Test backfilling a specific file:

```bash
python test_specific_backfill.py
```

### Single Symbol Test

Test backfilling a single symbol with limited rows:

```bash
python test_single_symbol.py
```

## Queries

### Check Backfilled Data

```sql
-- Count rows by symbol
SELECT symbol, COUNT(*) as row_count
FROM coinbase_source
GROUP BY symbol
ORDER BY row_count DESC;

-- Check data by timeframe
SELECT symbol, timeframe, COUNT(*) as row_count
FROM coinbase_source
GROUP BY symbol, timeframe
ORDER BY symbol, timeframe;

-- Query specific symbol
SELECT * FROM coinbase_source
WHERE symbol = 'BTC-USD'
ORDER BY timestamp
LIMIT 10;

-- Get date range
SELECT 
    symbol,
    MIN(timestamp) as start_date,
    MAX(timestamp) as end_date,
    COUNT(*) as row_count
FROM coinbase_source
GROUP BY symbol;
```

### Provenance Queries

```sql
-- Check provenance metadata
SELECT * FROM provenance_metadata
WHERE source_exchange = 'coinbase'
ORDER BY import_timestamp DESC;

-- Check data quality
SELECT 
    symbol,
    COUNT(*) as total_rows,
    MIN(timestamp) as min_time,
    MAX(timestamp) as max_time
FROM provenance_metadata p
JOIN coinbase_source c ON c.symbol = p.symbol
WHERE p.source_exchange = 'coinbase'
GROUP BY symbol;
```

## Error Handling

The agent handles:

1. **Missing files**: Logs but continues processing
2. **Empty data**: Skips empty feather files
3. **Database errors**: Continues with other symbols
4. **Schema mismatches**: Automatically adds missing columns with defaults
5. **Timestamp issues**: Handles different timestamp formats

## Performance Considerations

1. **Batch Insertion**: Processes data in batches for efficiency
2. **Indexing**: Creates indexes on symbol, timestamp, and timeframe
3. **Memory Management**: Processes large files in chunks
4. **Duplicate Handling**: Uses INSERT OR REPLACE to avoid duplicates

## Output

The agent generates:

1. **Console Output**: Progress and status messages
2. **Provenance Log**: JSON file at `hrm/data/coinbackbackfill_provenance_log.json`
3. **DuckDB Database**: Updated with Coinbase data
4. **Report**: Comprehensive summary of backfill operation

## Example Output

```
================================================================================
COINBASE BACKFILL AGENT REPORT
================================================================================

BACKFILL RESULTS:
  Total pairs backfilled: 4
  Total rows backfilled: 1500000
  Timeframes processed: 1m
  Failed pairs: 0

PROVENANCE TRACKING:
  Import timestamp: 2026-02-20T13:57:41.764857
  Total entries logged: 4
  DuckDB database: hrm/data/market.duckdb

DATA STRUCTURE:
  coinbase_source - Coinbase data with 48-column schema and provenance
  provenance_metadata - Complete provenance tracking
  market_data - Unified view of all data

QUERIES:
  SELECT * FROM provenance_metadata WHERE source_exchange = 'coinbase'
  SELECT * FROM coinbase_source WHERE symbol = 'BTC-USD'
  SELECT * FROM market_data WHERE symbol = 'BTC-USD' ORDER BY timestamp
  SELECT symbol, timeframe, COUNT(*) FROM coinbase_source GROUP BY symbol, timeframe

NEXT STEPS:
  1. Verify data quality with: SELECT symbol, COUNT(*) FROM coinbase_source GROUP BY symbol
  2. Check timeframes: SELECT symbol, timeframe, COUNT(*) FROM coinbase_source GROUP BY symbol, timeframe
  3. Query specific symbol: SELECT * FROM coinbase_source WHERE symbol = 'ETH-USD' ORDER BY timestamp
```

## Troubleshooting

### Issue: No data found in arrow directory

**Solution**: Check that feather files exist in `hrm/data/arrow/` with Coinbase-style naming (e.g., `BTC_USD.feather`, `ETH_USD.feather`)

### Issue: Database connection error

**Solution**: Ensure DuckStore is available and DuckDB file is accessible

### Issue: Missing columns in 48-column schema

**Solution**: The agent automatically adds missing columns with default values (0.0 for numeric, False for boolean)

### Issue: Large file processing is slow

**Solution**: The agent processes files in batches. For very large files, consider increasing the batch size in the code

## Integration with Existing Systems

This agent complements the existing `provenance_import.py` by:

1. **Coinbase Focus**: Handles only Coinbase data
2. **48-Column Schema**: Uses the full schema required for live agents
3. **Timeframe Support**: Explicitly handles different timeframes
4. **Provenance Integration**: Uses the same provenance tracking system

## Future Enhancements

1. **Parallel Processing**: Process multiple symbols simultaneously
2. **Gap Detection**: Identify and fill temporal gaps in existing data
3. **Data Validation**: Add data quality checks and validation
4. **Performance Metrics**: Track import speed and database performance
5. **Web Interface**: Create a simple web UI for monitoring backfill operations

## License

This tool is part of the MoneyFan project and follows the same license as the rest of the codebase.