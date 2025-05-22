# Moneyfan

A lightweight, performant Java library for handling 2D columnar data, specifically designed for financial time-series analysis and simulation agent feeds.

## Features

- Immutable data structures for thread safety
- Lazy evaluation for efficient processing of large datasets
- Memory-mapped I/O (ISAM-like) for high performance
- Fluent API for data transformations
- CSV import and export
- Java Records for concise, data-centric implementation

## Project Structure

The project is organized into a few core packages:

- `com.moneyfan.core`: Core data types (Pai2, Scalar, IOMemento, CellMeta)
- `com.moneyfan.grid`: Grid structures (Vect0r, Cell, RowVec, GridCursor)
- `com.moneyfan.io`: I/O utilities (CellDriver, FixedDriver, CSVCursorReader, ISAMReader, ISAMWriter)
- `com.moneyfan.binance`: Binance data integration example

## Quick Start

### Reading CSV Data

```java
// Define schema
List<Scalar> schema = List.of(
    Scalar.of(IOMemento.IO_INT, "id"),
    Scalar.of(IOMemento.IO_STRING_FIXED, "name", 20),
    Scalar.of(IOMemento.IO_DOUBLE, "value")
);

// Read CSV
GridCursor cursor = CSVCursorReader.readFromCSV(Path.of("data.csv"), schema, true);

// Use the cursor
System.out.println("Row count: " + cursor.rowCount());
```

### Processing Data

```java
// Select specific columns
GridCursor selectedData = cursor.select("id", "value");

// Filter rows
GridCursor filteredData = cursor.filter(row -> 
    ((Double) row.getValue(2)) > 100.0
);
```

### Persisting to ISAM Format

```java
// Write to ISAM format
ISAMWriter.writeGridCursor(cursor, Path.of("data.bin"), Path.of("data.meta"));

// Read from ISAM format
try (ISAMReader reader = new ISAMReader(Path.of("data.bin"), Path.of("data.meta"))) {
    GridCursor mmapCursor = reader.open();
    // Use memory-mapped data
}
```

### Binance Data Example

```java
// Process Binance kline data
BinanceIngest.processKlineData(
    Path.of("binance_btcusdt_1m.csv"),
    Path.of("btcusdt_1m.bin"),
    Path.of("btcusdt_1m.meta")
);

// Use in agent simulation
try (ISAMReader reader = new ISAMReader(
        Path.of("btcusdt_1m.bin"), 
        Path.of("btcusdt_1m.meta"))) {
    
    GridCursor cursor = reader.open();
    BinanceIngest.AgentDataProvider provider = 
        new BinanceIngest.BinanceAgentDataProvider(cursor);
    
    // Feed data to agents
    RowVec dataPoint;
    while ((dataPoint = provider.getNextDataPoint()) != null) {
        // Process data point in agent
    }
}
```

## Building the Project

```bash
mvn clean install
```

## Requirements

- Java 17 or higher
- Maven 3.6 or higher