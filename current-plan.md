# Moneyfan Columnar Java Project

## Project Vision

To create a highly performant, memory-efficient, and developer-friendly Java library for handling 2D columnar data, specifically tailored for financial time-series analysis and simulation agent feeds. This project draws inspiration from the design principles of the Kotlin `columnar` toolkit (immutable data structures, lazy evaluation, ISAM-like storage) but aims for a streamlined Java implementation using Records as lightweight value classes and a powerful, concise DSL.

The initial focus is on robustly ingesting Binance market data (simulating the `acapulco` data pipeline) and making it available in an efficient columnar format for consumption by simulator agents (akin to the needs of `control`).

## Core Principles

1.  **Immutability:** All core data structures (Records representing pairs, vectors, rows, grids) are immutable to ensure thread safety and predictable state.
2.  **Java Records as Value Types:** Leverage Java Records for concise, data-centric "inline value classes." Design with an eye towards future Project Valhalla primitive class benefits.
3.  **Static Factories:** Prefer static factory methods for instantiation to promote a clear DSL and good IntelliJ autocompletion.
4.  **Lazy Evaluation:** Operations on data grids (select, filter, map) should, where possible, return new lazy views rather than immediately processing all data.
5.  **Memory-Mapped IO (ISAM-like):** Data persistence and retrieval will be based on a simple, fixed-record-length binary format with associated metadata, enabling efficient mmap access for large datasets.
6.  **Functional DSL:** Provide a fluent and expressive DSL for columnar operations, using lambdas (functional interfaces) extensively.
7.  **Compact & Utilitarian:** Focus on a minimal set of powerful primitives that can be composed effectively, avoiding unnecessary code and complexity.
8.  **Enum-Driven Metadata:** Utilize enums (e.g., `IOMemento`) extensively for type and metadata management.

## Architectural Overview (Target Modules)

*   **`moneyfan-core`**:
    *   `Pai2<F,S>`: Fundamental immutable pair record.
    *   `IOMemento`: Enum for data types and their binary characteristics.
    *   `Scalar`: Record for column type and name.
    *   `CellMeta`: Record for cell metadata provider (Supplier of Scalar).
*   **`moneyfan-grid`**:
    *   `Vect0r<T>`: Lazy, immutable vector record.
    *   `Cell`: Record for cell value and its `CellMeta`.
    *   `RowVec`: Record representing a row as a `Vect0r<Cell>`.
    *   `GridCursor`: Record representing the 2D grid as a `Vect0r<RowVec>`.
*   **`moneyfan-io`**:
    *   `CellDriver<Buffer, Value>`: Interface for reading/writing cell values.
    *   `FixedDriver<Value>`: `CellDriver` for fixed-size primitive types (using `java.nio.ByteBuffer`).
    *   `CSVCursorReader`: Utility to read CSV files into an in-memory `GridCursor`.
    *   `ISAMMeta`: Record/class to represent/parse/write ISAM metadata.
    *   `ISAMReader`: Reads ISAM data files (binary + meta) into a lazy, mmap-backed `GridCursor`.
    *   `ISAMWriter`: Writes a `GridCursor` to ISAM format.
*   **`moneyfan-dsl`**: (Methods primarily on `GridCursor`)
    *   Columnar expression operations (select, filter, map, etc.).
*   **`moneyfan-integrations` (or `moneyfan-binance-ingest`)**:
    *   Specific logic to replicate the `acapulco` Binance data pipeline using the new `moneyfan` components.
    *   Adapters to feed data into a simplified simulator agent interface.

## Project Roadmap & Prongs of Attack

This roadmap outlines distinct phases, allowing for focused development and iterative progress.

### Phase 1: Foundational Java Types & Core Grid Structures

*   **Objective:** Establish the immutable, record-based building blocks.
*   **Tasks:**
    1.  **Implement `moneyfan-core`:**
        *   `Pai2<F,S>` record with `Pai2.of(F,S)` factory.
        *   `IOMemento` enum (covering all necessary types from your Kotlin project, e.g., `IoInt`, `IoLong`, `IoDouble`, `IoLocalDate`, `IoInstant`, `IoStringFixed`).
        *   `Scalar(IOMemento type, String name)` record.
        *   `CellMeta(Supplier<Scalar> provider)` record.
    2.  **Implement `moneyfan-grid` (core parts):**
        *   `Vect0r<T>(int size, IntFunction<T> accessor)` record with `Vect0r.of(...)`, `Vect0r.fromList(...)` factories. Ensure basic `get(int)` functionality.
        *   `Cell(Object value, CellMeta meta)` record.
        *   `RowVec(Vect0r<Cell> cells)` record.
        *   `GridCursor(Vect0r<RowVec> rows)` record with `rowCount()`, `columnCount()`, `getRow(int)`, `getScalars()`.
*   **Testing:** Unit tests for immutability, factory creation, and basic accessors for all records.
*   **Team Prong:** Core Java developers focused on data structures.

### Phase 2: ISAM Implementation & CSV Ingest

*   **Objective:** Enable persistent, mmap-able storage and the first step of the data pipeline (CSV reading).
*   **Tasks:**
    1.  **Implement `moneyfan-io` (Cell Drivers & ISAM Meta):**
        *   `CellDriver` interface.
        *   `FixedDriver<Value>` record implementing `CellDriver` for core `IOMemento` types (int, long, double, date, instant). Include static map of drivers. Special attention to `IO_STRING_FIXED` where length comes from metadata.
        *   `ISAMMeta` class/record:
            *   Define structure (column names, types, offsets, record length, string fixed lengths).
            *   Implement parsing from a `.meta` file.
            *   Implement writing to a `.meta` file.
    2.  **Implement `ISAMReader`:**
        *   Constructor takes `Path dataPath`, `Path metaPath`.
        *   Parses `.meta` file using `ISAMMeta`.
        *   Uses `java.nio.FileChannel.map(MapMode.READ_ONLY, ...)` to mmap the data file.
            *   Start with mapping the whole file; windowing for >2GB files can be a later optimization.
        *   `open()` method returns a `GridCursor` where:
            *   `RowVec` accessor reads a record-sized slice from `MappedByteBuffer`.
            *   `Cell` accessor uses the appropriate `CellDriver` and column offset to read the value from the row's `ByteBuffer` slice. This must be *lazy*.
    3.  **Implement `CSVCursorReader`:**
        *   Takes `Path csvPath`, list of `Scalar` (or `IOMemento`s and names) for schema.
        *   Reads CSV into an in-memory `GridCursor` (can use `ArrayList` internally to build `Vect0r.fromList`). This is an eager load, but necessary before ISAM persistence.
    4.  **Implement `ISAMWriter`:**
        *   Takes `GridCursor` and `Path` for output.
        *   Writes `.meta` file using `ISAMMeta` based on `GridCursor.getScalars()`.
        *   Iterates `GridCursor`, writes each row to a `ByteBuffer`, then to `FileChannel` (not necessarily mmap for writing initially, `FileChannel.write()` is fine).
*   **Testing:**
    *   Unit tests for `CSVCursorReader` with sample CSVs.
    *   Write a `GridCursor` (from CSV or manually created) using `ISAMWriter`.
    *   Read it back with `ISAMReader` and verify data integrity and schema.
    *   Test with various `IOMemento` types.
*   **Team Prong:** IO specialists, Java NIO experts.

### Phase 3: Basic Columnar DSL & Lazy Expressions

*   **Objective:** Provide initial data manipulation capabilities on `GridCursor`.
*   **Tasks (add methods to `GridCursor` record):**
    1.  **`select(String... columnNames)`:**
        *   Returns a new `GridCursor`.
        *   The new `GridCursor`'s `Vect0r<RowVec>` accessor maps row requests to the original cursor.
        *   The new `RowVec`'s `Vect0r<Cell>` accessor maps column requests to the selected columns of the original row. This is lazy.
    2.  **`mapColumn(String columnName, IOMemento newType, Function<Object, Object> transform)`:**
        *   Returns a new `GridCursor` where one column is transformed.
        *   Lazy: the transform `Function` is applied only when the cell value is accessed.
    3.  **`filter(Predicate<RowVec> predicate)`:**
        *   Returns a new `GridCursor`. This is the trickiest for pure mmap laziness without an intermediate index.
        *   *Initial approach:* Could be semi-lazy by identifying row indices that match, then creating a new `Vect0r` accessor that only serves these indices.
        *   *Advanced approach (later):* BitSet for filtered rows, or JIT-compiled predicates.
*   **Testing:** Unit tests for each DSL operation, verifying lazy behavior where applicable and correct data transformation/selection.
*   **Team Prong:** API/DSL designers, developers comfortable with functional programming in Java.

### Phase 4: Binance Data Ingest Integration (`acapulco`/`control` Port/Adaptation)

*   **Objective:** Replicate the core data pipeline for Binance data and prepare it for simulator agent consumption.
*   **Tasks:**
    1.  **Analyze `acapulco`'s Binance data download scripts/logic:**
        *   Understand the output format (CSV structure, column names, types).
        *   If shell scripts are essential, keep them and ensure their CSV output is standardized.
    2.  **Create `moneyfan-binance-ingest` module (or similar):**
        *   **Binance CSV Schema Definition:** Define `Scalar` lists or `IOMemento` arrays corresponding to the Binance CSV kline data.
        *   **Ingest Process:**
            *   A Java process that:
                *   (Optional) Invokes existing shell scripts to download data (if they can't be easily ported to Java).
                *   Uses `CSVCursorReader` to read the downloaded CSVs into `GridCursor`s.
                *   Applies any necessary initial transformations (e.g., date parsing, simple type conversions if not handled by CSV reader schema) using the new DSL.
                *   Uses `ISAMWriter` to persist the processed `GridCursor`s into the ISAM format.
    3.  **Define Simulator Agent Feed Interface:**
        *   What data does the `control` module's simulator agent *need* per time step? Is it a full `RowVec`, specific columns, or a transformed view?
        *   Create a simple Java interface for this (e.g., `AgentDataProvider` with a `getNextDataPoint() -> AgentDataRecord`).
    4.  **Agent Data Provider Implementation:**
        *   A class that uses `ISAMReader` to open the relevant ISAM files.
        *   Its `getNextDataPoint()` method would advance its internal row pointer in the `GridCursor` and return the necessary data, possibly transformed using the DSL.
*   **Testing:**
    *   End-to-end test: Download sample Binance CSV, process with `moneyfan-binance-ingest`, write to ISAM.
    *   Read from ISAM using the `AgentDataProvider` and verify data matches expectations.
*   **Team Prong:** System integrators, developers familiar with `acapulco` and `control` logic.

### Phase 5: Performance Tuning, Advanced DSL, and Refinements

*   **Objective:** Optimize, expand capabilities, and harden the library.
*   **Tasks:**
    1.  **Benchmarking:** Compare performance (read/write speed, memory usage) against:
        *   Original Kotlin `columnar` (for relevant operations).
        *   Other Java columnar libraries (if applicable, e.g., Apache Arrow's Java bindings for specific use cases).
    2.  **Mmap Windowing:** Implement for `ISAMReader` if files routinely exceed 2GB and whole-file mapping becomes a bottleneck.
    3.  **Advanced DSL Features:**
        *   `groupBy(...)` and aggregations (e.g., `sum`, `avg` over groups) – these are often eager.
        *   `sortBy(...)` – likely eager or requires index creation.
        *   `join(...)` operations between `GridCursor`s.
    4.  **"VTable of Lambdas" for IOMemento:** Refine `CellDriver` and its usage. Ensure efficient dispatch for read/write operations based on `IOMemento`. This is largely covered by the `FixedDriver.MAPPED_DRIVERS` approach.
    5.  **Error Handling & Robustness:** Comprehensive exception handling, validation of meta files, etc.
    6.  **Investigate JIT Optimizations:** Profile and see how Java Records are being handled, especially with future Valhalla features in mind.
*   **Team Prong:** Performance engineers, senior developers.

## Key Design Considerations & Challenges

*   **Mmap Laziness vs. Operational Complexity:** True laziness for operations like `filter` or `sortBy` on mmap'd data without loading everything into memory often requires creating intermediate index structures or more complex accessor logic. A pragmatic balance will be needed.
*   **String Handling in Fixed-Record ISAM:** `IO_STRING_FIXED` requires careful management of fixed lengths in metadata and during read/write. Variable-length strings are much harder in a simple fixed-record ISAM and would necessitate a different storage strategy (e.g., separate string pool, dictionary encoding).
*   **Concurrency:** While immutability helps, `FileChannel` and `MappedByteBuffer` have their own concurrency considerations if multiple threads were to access the *same* `ISAMReader` instance (though the design leans towards one reader per "view"). The primary goal here is single-threaded performance for the agent feed.
*   **DSL Expressiveness vs. Java Verbosity:** Crafting a DSL that feels "Go-like" or as expressive as Kotlin's in pure Java is challenging. Judicious use of static imports, fluent interfaces, and well-named factory methods will be key.
*   **Avoiding Kotlin's "Error-Prone" Evolution Path:** This means being very deliberate about the scope, sticking to simpler Java features initially, and thoroughly testing each component. The complexity in the Kotlin project might have stemmed from trying to achieve too much too quickly or from very advanced generic programming.

## Getting Started

1.  Set up the Maven project structure (`moneyfan-core`, `moneyfan-grid`, `moneyfan-io`).
2.  Begin with Phase 1, implementing the core record types.
3.  Progress through the phases, prioritizing the ISAM read/write and CSV ingest for the Binance data pipeline.

This roadmap aims for a focused, iterative development of the "moneyfan" library, directly addressing the need for a robust and performant Java-based solution for the specified Binance data ingest and simulation feed use case.