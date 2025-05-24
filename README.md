# DSEL & bbcursive Project

This project contains a parent POM and two primary modules: `bbcursive` and `bikeshed`.
It aims to provide a robust Data Science Expression Language (DSEL) built upon an efficient,
zero-copy `ByteBuffer` parsing combinator library.

## Modules

### 1. `bbcursive`

*   **Purpose:** A minimal, modern, zero-copy `ByteBuffer` parsing combinator library. It provides foundational interfaces, enums, and helper classes for building efficient parsers that operate directly on `ByteBuffer`s. The emphasis is on functional composition, clarity, and performance by avoiding unnecessary data copying.
*   **Core Concepts:**
    *   `Cursive`: The primary functional interface for a parsing operation (`UnaryOperator<ByteBuffer>` or a more advanced `TypedParser<T>`).
    *   `BBAtom`: Enum or static factories for basic byte/char/string literal matching, view manipulation (slice, duplicate, position, limit), and other primitive parsing tasks.
    *   `BBCombinator`: Static factories for higher-order parsing functions (e.g., `sequence`, `choice`, `optional`, `many`).
    *   `ParseResult<T>`: A wrapper for results from typed parsers, carrying the parsed value and the remaining buffer state.
*   **Key Features:**
    *   Zero-copy operations on `ByteBuffer`s.
    *   Functional, composable parser design.
    *   Extensible for defining custom parsing rules.
    *   Minimal dependencies.

### 2. `bikeshed` (DSEL Core - formerly `moneyfan.dsel`)

*   **Purpose:** The core implementation of the Data Science Expression Language (DSEL). It provides high-level data structures (`Series`, `Cursor`, `RowVec`, `Join`), a type system (`TypeMemento`), and utilities for data manipulation, CSV processing, and ISAM-like fixed-format file I/O.
*   **Integration with `bbcursive`:**
    *   The ISAM reading and writing functionalities within `bikeshed` are (or will be) refactored to use `bbcursive` for all low-level `ByteBuffer` parsing and manipulation. This replaces manual byte operations with structured, grammar-based parsing rules defined using `bbcursive`.
    *   CSV parsing may also leverage `bbcursive` for enhanced performance and robustness.
    *   Type deduction mechanisms will operate on `ByteBuffer` views provided by `bbcursive`-powered parsers.
*   **Key Features:**
    *   Expressive data manipulation abstractions.
    *   Lazy evaluation where appropriate (e.g., `Cursor`, `Series` operations).
    *   Efficient ISAM-like storage and retrieval, built on `bbcursive`.
    *   CSV utilities with type deduction.

## Vision

The combination of `bbcursive`'s efficient byte-level parsing capabilities and `bikeshed`'s high-level data abstractions aims to create a powerful and robust toolkit for data processing. This architecture promotes:

*   **Performance:** Through zero-copy `ByteBuffer` operations and efficient parsing.
*   **Robustness:** By defining binary formats and parsing logic declaratively as grammars, reducing errors from manual byte manipulation.
*   **Maintainability:** Clear separation of concerns between low-level parsing (`bbcursive`) and high-level data logic (`bikeshed`).
*   **Extensibility:** Easier to add support for new binary formats or parsing requirements by defining new `bbcursive` rules.

## Building the Project

This is a Maven project. To build:

