# MoneyFan Columnar Java Project

MoneyFan is a lightweight, immutable, columnar‐data toolkit designed for high-performance financial time-series work in pure Java.

## Highlights

* **Immutable Records** – All core types (`Pai2`, `Vect0r`, `RowVec`, `GridCursor`, …) are Java Records.
* **Lazy Columnar Views** – `select`, `filter`, `mapColumn`, `sortBy`, `groupBy`, joins and aggregations build views instead of materialising data.
* **Memory-mapped ISAM** – Huge binary files (> 2 GB) are memory-mapped in windowed slices for O(1) random row access.
* **DSL** – Fluent API for common transformations (`grid.select("price")…`).
* **JMH Benchmarks** – `GridCursorBenchmark` & `CSVvsISAMReadBenchmark` measure throughput and latency.
* **Binance Ingest** – Download daily klines CSV → ISAM in one call.

## Modules

```
core       – basic immutable metadata types
grid       – 2-D data structures & lazy DSL
io         – Cell drivers, CSV reader, ISAM reader/writer
bench      – JMH micro-benchmarks (src/jmh/java)
binance    – CSV downloader & converter helper
```

## Quick Start

```bash
mvn test            # all unit tests (JUnit 5)
# Run micro-benchmarks (will compile & execute JMH):
mvn clean package -DskipTests && java -jar target/benchmarks.jar
```

### Reading a CSV
```java
List<Scalar> schema = List.of(
    Scalar.of(IOMemento.IO_INT, "id"),
    Scalar.of(IOMemento.IO_DOUBLE, "price")
);
GridCursor grid = CSVCursorReader.read(Path.of("prices.csv"), schema);
```

### Persisting to ISAM & Re-opening lazily
```java
ISAMWriter.write(grid, Path.of("prices.bin"));
try (ISAMReader reader = new ISAMReader(Path.of("prices.bin"))) {
    GridCursor lazy = reader.open();
    double p0 = (Double) lazy.getRow(0).get(1).value();
}
```

### Binance Ingest
```java
Path csv = BinanceIngest.downloadDailyKlines("BTCUSDT", "1h", LocalDate.of(2023,1,1), Path.of("/tmp"));
BinanceIngest.convertCsvToIsam(csv, Path.of("btc-1h.bin"));
```

## Future Work
* Additional aggregation operators (`avg`, `min`, `max`).
* Predicate push-down for `ISAMReader` filters.
* Parquet / Arrow export.
* Parallel pipeline utilities.