# DSEL Bbcursive Framework

This project implements a high-performance data processing framework using a Domain-Specific Embedded Language (DSEL) named `bbcursive`. It is designed to handle time-series and large-scale data challenges by leveraging `ByteBuffer`-centric, cursor-based operations.

## Core DSEL Philosophy and Design Principles:

1.  **Immutability and Compositional Purity:**
    *   The foundational primitive is the `Join<F, S>` interface (implemented as `com.example.dsel.bikeshed.core.Join`), an immutable 2-tuple.
    *   All operations on `Join` or its derived types (`Series`, `Cursor`, `RowVec`, `Twin`) produce *new* instances rather than modifying existing ones.
    *   This promotes predictable behavior, simplifies concurrency, and enables powerful optimization (e.g., memoization, lazy evaluation).

2.  **Kotlin-Inspired Meta-programming in Java:**
    *   **Operator Overloading via Convention:** Methods like `plus`, `minus`, `get`, `div` are implemented where semantically appropriate for `Join` and `Series` to provide concise, infix-style operations.
    *   **Type Aliases (`typealias` equivalent):** Strong naming conventions and Javadoc comments are used (e.g., `ColumnMeta` as `Join<String, TypeMemento>`) to create meaningful, short-hand names for complex generic types.
    *   **Extension Functions (via Static Utilities/Wrappers):** Functionality is enriched via static methods in utility enums (e.g., `D.java`) that act as "extension functions." For instance, `D.jn(a,b)` is used for `Join.of(a,b)`.
    *   **Glyph-based Shorthands (IDE Optimized):** API elements use short, memorable "glyphs" or abbreviations (e.g., `jn` for `Join`, `α` for `map` or `transform`, `▶` for `iterable view`) to optimize for IDE code completion.

3.  **Encapsulation via Enums:**
    *   Related functionality and constants are grouped within `enum` types (e.g., `D.java`, `IOMemento`, `BBAtom`, `BBCombinator`). This centralizes operations, provides natural namespaces, and promotes discoverability.

4.  **Lambda Capture Minimization & Function References:**
    *   User-facing APIs primarily use function references (`ClassName::methodName`) or simple lambdas that *only* capture `Join` instances or the results of `Join`-based operations.
    *   Complex lambdas capturing mutable external state are avoided to maintain purity and simplify reasoning about data flow.
    *   Ternary operators are preferred for concise conditional logic where readability is maintained.

## Project Structure:

The Maven project is organized into a parent POM (`dsel-bbcursive-parent`) and two key modules:

*   **`bbcursive` (Low-Level `ByteBuffer` Operations):**
    *   **Purpose:** A modern, zero-copy `ByteBuffer` parsing combinator library. It provides foundational interfaces (`com.example.dsel.bbcursive.core.Cursive`) and utilities (`com.example.dsel.bbcursive.BBAtom`, `com.example.dsel.bbcursive.BBCombinator`, `com.example.dsel.bbcursive.util.ByteParsers`) for highly efficient, direct `ByteBuffer` manipulation with a functional composition emphasis.
    *   **Key Features:** Native-aligned `ByteBuffer` operations, functional parser combinators, extensible for custom parsing rules, minimal dependencies.

*   **`bikeshed` (DSEL Core):**
    *   **Purpose:** The high-level DSEL implementation for data structures (`com.example.dsel.bikeshed.core.Series`, `com.example.dsel.bikeshed.dsel.Cursor`, `com.example.dsel.bikeshed.dsel.RowVec`, `com.example.dsel.bikeshed.core.Join`), type system (`com.example.dsel.bikeshed.types.TypeMemento`, `com.example.dsel.bikeshed.types.IOMemento`, `com.example.dsel.bikeshed.types.ColumnMeta`), and data manipulation.
    *   **Integration with `bbcursive`:** All low-level data access (ISAM, CSV) leverages `bbcursive` for direct `ByteBuffer` parsing, crucial for memory-mapped file alignment and zero-copy operations.
    *   **Key Features:**
        *   **`Join<F, S>`:** The core immutable 2-tuple foundation, with compositional transformations.
        *   **`Series<T>`:** A cursor-based collection optimized for ISAM data access patterns (e.g., `get(index)`, `slice(range)`), supporting memory-mapped files (mmap) for efficient time-series data.
        *   **`RowVec`:** Represents a row of data (`Series<Join<Any?, () -> ColumnMeta>>`) with associated metadata.
        *   **`Cursor`:** The primary interface for tabular data (`Series<RowVec>`), providing columnar abstraction.
        *   **`Twin<T>`:** A specialized `Join<T, T>` for symmetric pairs.
        *   **`TypeMemento` & `IOMemento`:** A robust type system defining data types and their serialization/deserialization strategies for ISAM. `IOMemento` entries explicitly specify fixed `networkSize` (bytes) or `null` for variable types requiring external configuration.
        *   **ISAM (`com.example.dsel.bikeshed.isam.IsamDataFile`, `com.example.dsel.bikeshed.isam.IsamMetaFileReader`, `com.example.dsel.bikeshed.isam.WireProto`):** Implements structured, fixed-format file I/O using `bbcursive` for byte-level parsing and mmap for zero-copy.
        *   **CSV Processing (`com.example.dsel.bikeshed.csv.CsvProcessor`):** Provides utilities for CSV ingestion, leveraging `bbcursive` for parsing.

## Trading Simulator & DSEL Integration:

The framework directly supports a concurrent trading simulator:

1.  **Binance Archive Ingestion:**
    *   The `com.example.dsel.bikeshed.trading.MarketDataSource` is designed to ingest large Binance historical data into `ISAM` structures. `ISAM`'s fixed-format, memory-mapped nature, combined with `Series` and `Cursor` APIs, ensures efficient storage and retrieval of time-series data. `bbcursive` handles the low-level parsing of this data into ISAM records.

2.  **Concurrent Trade Agents on Shared Timeline:**
    *   The DSEL supports multiple, independent `com.example.dsel.bikeshed.trading.AgentInterface` implementations operating on a *shared, immutable timeline* (`Series<MarketTick>` or `Cursor` of ticks).
    *   **Data Access:** Agents access market data via `Series` and `Cursor` interfaces, backed by mmap/ISAM, ensuring zero-copy access and minimizing contention.
    *   **Concurrency Model:** While `java.util.concurrent` is used for coordinating agents (e.g., `ExecutorService`), the DSEL's immutable `Join`-based records inherently reduce the need for complex locking mechanisms on data *access*. Agent actions and wallet state updates occur on agent-local or transaction-specific mutable state, which is then integrated compositionally into new immutable states.
    *   **Time-series Navigation:** `Cursor` operations (e.g., `at`, `row`, `get(range)`) enable agents to efficiently navigate and slice historical data based on their lookback periods.

3.  **Reward-Based Agent Visibility:**
    *   DSEL primitives (`Join`, `Series`, `Cursor`) can construct and evaluate predicates on agent performance records, dynamically adjusting data visibility. This might involve using `D.filter` on `Cursor` views based on `Series<Boolean>` (predicate results) derived from agent performance.

## Coding Style Mandates:

*   **Enum for Grouping:** Related functions and constants are grouped within `enum` bodies (e.g., `D.java`, `TypeMemento`).
*   **Lambdas Capture Only Joins:** Lambda captures are restricted to `Join` instances or `Join`-based computation results for functional purity.
*   **Function References for Brevity:** `ClassName::methodName` is preferred for passing functions.
*   **Simple Lambdas:** Lambdas are kept concise and single-purpose.
*   **Syntax Shortening:** Ternary operators (`condition ? true_val : false_val`) are used for concise conditional logic where clarity is maintained.
