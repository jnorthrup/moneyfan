As the lead architect for a high-performance data processing framework, your task is to design and implement a Maven Java project that introduces a sophisticated Domain-Specific Embedded Language (DSEL). This DSEL, named `bbcursive` for its `ByteBuffer`-centric, cursor-based operations, is destined to supersede conventional data manipulation libraries like Pandas, particularly for time-series and large-scale data challenges.

**Core DSEL Philosophy and Design Principles:**

1.  **Immutability and Compositional Purity:** The DSEL's foundational primitive is the `Join<F, S>` interface, an immutable 2-tuple. All operations on `Join` or its derived types (like `Series`, `Cursor`, `RowVec`, `Twin`) must be compositional, meaning they produce *new* instances rather than modifying existing ones. This immutability promotes predictable behavior, simplifies concurrency, and enables powerful optimization techniques (e.g., memoization, lazy evaluation).

2.  **Kotlin-Inspired Meta-programming in Java:** Leverage Java's modern features to achieve a meta-programming style akin to Kotlin. This includes:
    *   **Operator Overloading via Convention:** Implement methods like `plus`, `minus`, `get`, `div` (where semantically appropriate) to provide concise, infix-style operations for `Join` and `Series`.
    *   **Type Aliases (`typealias` equivalent):** Use `typedef` (or similar Javadoc/naming conventions for clarity if `typedef` isn't directly supported) to create meaningful, short-hand names for complex generic types, e.g., `typealias Join<F, S> = Pair<F, S>`. (Note: Actual `typealias` is Kotlin-specific; in Java, this means strong naming conventions).
    *   **Extension Functions (via Static Utilities/Wrappers):** Employ static utility classes or enum methods that act as "extension functions" to enrich existing types without modifying them. For instance, `D.jn(a,b)` for `Join.of(a,b)`.
    *   **Glyph-based Shorthands (IDE Optimized):** Design API elements with short, memorable "glyphs" or abbreviations (e.g., `jn` for `Join`, `α` for `map` or `transform`, `▶` for `iterator` or `iterable view`). These must be unique and optimized for IntelliJ IDEA's code completion, typically by using consistent prefixes or suffixes to avoid clashes and provide quick access.

3.  **Encapsulation via Enums:** Condense related functionality and constants into `enum` types. This strategy centralizes operations, provides a natural namespace, and promotes discoverability. For example, `D.java` will become an omnibus enum for common DSEL operations. Similarly, `TypeMemento` and `IOMemento` already demonstrate this pattern for type information and I/O strategies.

4.  **Lambda Capture Minimization & Function References:** User-facing APIs should primarily use function references (`ClassName::methodName`) or simple lambdas that *only* capture `Join` instances or the results of `Join`-based operations. Avoid complex lambda expressions that capture mutable external state to maintain purity and simplify reasoning about data flow. Ternary operators should be preferred for concise conditional logic where readability is not compromised.

**Project Structure and Module Breakdown:**

Your Maven project will comprise a `dsel-bbcursive-parent` POM and two key modules:

*   **`bbcursive` (Low-Level `ByteBuffer` Operations):**
    *   **Purpose:** A modern, zero-copy `ByteBuffer` parsing combinator library. It provides the foundational interfaces and utilities for highly efficient, direct `ByteBuffer` manipulation, emphasizing functional composition.
    *   **Core Interfaces/Enums:** `Cursive` (likely `UnaryOperator<ByteBuffer>` or a more generic `Parser<T, ByteBuffer>`), `BBAtom` (for primitive byte/char/string literal matching, buffer view manipulation: `slice`, `duplicate`, `position`, `limit`), `BBCombinator` (for higher-order parsing functions: `sequence`, `choice`, `optional`, `many`).
    *   **Key Features:** Native-aligned `ByteBuffer` operations, functional parser combinators, extensible for custom parsing rules, minimal dependencies.

*   **`bikeshed` (DSEL Core - formerly `moneyfan.dsel`):**
    *   **Purpose:** The high-level DSEL implementation for data structures (`Series`, `Cursor`, `RowVec`, `Join`), type system (`TypeMemento`, `IOMemento`), and data manipulation.
    *   **Integration with `bbcursive`:** All low-level data access (ISAM, CSV) will be refactored to extensively utilize `bbcursive` for direct `ByteBuffer` parsing, replacing manual byte operations with composable grammar rules. This is crucial for achieving memory-mapped file alignment.
    *   **Key Features:**
        *   **`Join<F, S>`:** The immutable 2-tuple foundation. Provide methods like `mapFst`, `mapSnd`, `mapBoth`, `swap` for compositional transformations. Ensure `Join` instances are the primary "record" type throughout the DSEL.
        *   **`Series<T>`:** A `Join<Integer, Function<Integer, T>>` subtype, acting as a cursor-based collection. It provides methods optimized for ISAM data access patterns (e.g., `get(index)`, `slice(range)`), supporting memory-mapped files (mmap) for efficient time-series data access. Emphasize *phased adoption* where `Series` replaces `List`-based workflows.
        *   **`RowVec`:** A `Series<Join<Any?, () -> ColumnMeta>>` (or similar), representing a row of data with associated metadata. This will encapsulate a record's values alongside their type and name.
        *   **`Cursor`:** A `Series<RowVec>`, representing a collection of `RowVec`s, providing columnar abstraction. This is the primary interface for tabular data.
        *   **`Twin<T>`:** A specialized `Join<T, T>` for symmetric pairs.
        *   **`TypeMemento` & `IOMemento`:** A robust type system defining data types and their corresponding serialization/deserialization strategies (encoders/decoders) for ISAM integration. `IOMemento` entries must explicitly specify fixed-size `networkSize` (in bytes) where applicable for ISAM alignment (e.g., `IoInt` as 4 bytes, `IoLong` as 8 bytes, `IoInstant` as 12 bytes). Variable-length types (e.g., `IoString`) will require a `Map<String, Int>` for explicit length configuration during `IsamMetaFileReader.write`.
        *   **ISAM:** Implement `IsamDataFile` and `IsamMetaFileReader` for structured, fixed-format file I/O, directly leveraging `bbcursive` for byte-level parsing. Ensure `ISAM` data files align with mmap for zero-copy operations.
        *   **CSV Processing:** Provide utilities for CSV ingestion, potentially leveraging `bbcursive` for parsing and `TypeEvidence` for type deduction.

**Trading Simulator Gap Analysis & DSEL Integration:**

Address the following requirements for a concurrent trading simulator, demonstrating how the `bbcursive` DSEL directly supports them:

1.  **Binance Archive Ingestion:** Design the `MarketDataSource` (in `acapulco`) to ingest large Binance historical data archives directly into `ISAM` structures. This implies `ISAM`'s ability to efficiently store and retrieve time-series data with high throughput, using `Series` and `Cursor` as the primary data access APIs. The `bbcursive` library will handle the low-level parsing of compressed or raw binary archive data.

2.  **Concurrent Trade Agents on Shared Timeline:** The DSEL must support multiple, independent `AgentInterface` implementations operating on a *shared, immutable timeline* (`Series<MarketTick>` or `Cursor` of ticks).
    *   **Data Access:** Agents access market data primarily through `Series` and `Cursor` interfaces, which are backed by mmap/ISAM. This ensures zero-copy access to shared data and minimizes contention.
    *   **Concurrency Model:** Leverage Java's `java.util.concurrent` primitives where necessary, but emphasize that the DSEL's immutable `Join`-based records inherently reduce the need for complex locking mechanisms on data *access*. Mutations (e.g., agent actions, wallet state updates) will occur on agent-local or transaction-specific mutable state, which then gets integrated compositionally.
    *   **Time-series Navigation:** `Cursor` operations (e.g., `at`, `row`, `get(range)`) are crucial for agents to efficiently navigate and slice historical data based on their internal lookback periods.

3.  **Reward-Based Agent Visibility:** Implement agent visibility rules based on dynamic "rewards" or performance metrics. This implies that DSEL primitives (`Join`, `Series`, `Cursor`) can be used to construct and evaluate predicates on agent performance records, dynamically adjusting which data (e.g., specific asset pairs, time periods) is visible or accessible to an agent. This might involve `Series<Boolean>` or similar predicate results to filter `Cursor` views.

**Coding Style Mandates:**

*   **Enum for Grouping:** Group related functions and constants within `enum` bodies or companion objects (for Kotlin interop) to create logical namespaces (e.g., `D.java`, `TypeMemento`).
*   **Lambdas Capture Only Joins:** Restrict lambda captures to `Join` instances or the results of `Join`-based computations to maintain functional purity and simplify dependency analysis.
*   **Function References for Brevity:** Prefer `ClassName::methodName` for passing functions as arguments.
*   **Simple Lambdas:** Keep lambda expressions concise and single-purpose.
*   **Syntax Shortening:** Aggressively apply ternary operators (`condition ? true_val : false_val`) for conditional assignments and expressions where it improves conciseness without sacrificing clarity.

**Initial D.java (Omnibus Operations) Definition:**

```java
// Example structure for D.java
package com.yourdomain.bikeshed.dsel;

import com.yourdomain.bikeshed.core.Join;
import com.yourdomain.bikeshed.core.Series;
// ... other necessary imports

public enum D {
    // This enum will contain static-like methods acting as global DSEL operations.
    // Example:
    CREATE_JOIN; // Placeholder for constructor or initial state

    public static <F, S> Join<F, S> jn(F f, S s) {
        return Join.of(f, s); // Using a hypothetical Join.of factory method
    }

    public static <T> Series<T> sr(int size, java.util.function.IntFunction<T> provider) {
        // Factory for Series
        // ... implementation using Join<Integer, Function<Integer, T>>
        return null;
    }

    // Add other common operations here, e.g., mapping, filtering, reducing on Series/Joins
    // public static <F, S, R> Join<R, S> mapFst(Join<F, S> join, Function<F, R> mapper) { ... }
    // public static <F, S, R> Join<F, R> mapSnd(Join<F, S> join, Function<S, R> mapper) { ... }
    // public static <F, S> Join<S, F> swap(Join<F, S> join) { ... }
    // public static <T> Series<T> filter(Series<T> series, Predicate<T> predicate) { ... }
    // public static <T, R> Series<R> map(Series<T> series, Function<T, R> mapper) { ... }
    // ... etc.

}