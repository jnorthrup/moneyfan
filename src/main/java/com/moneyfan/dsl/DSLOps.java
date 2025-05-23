package com.moneyfan.dsl;

import com.moneyfan.dsl.row.Row;
import java.util.function.Function;

/**
 * An enum to house DSL operations and factory methods,
 * adhering to the "enums for bags of code elements" principle.
 * Provides shorthands for common data access patterns.
 */
public enum DSLOps {
    /** Singleton instance for accessing DSL operations. */
    Ops; // Using "Ops" as a short, meaningful name for the singleton

    /**
     * Creates a unary operator to project a column by name from a Row.
     * Shorthand: col(name)
     * @param columnName The name of the column.
     * @return A function that extracts the column value from a Row.
     */
    public <T> Function<Row, T> col(String columnName) {
        return r -> r.get(columnName);
    }

    /**
     * Creates a unary operator to project a column by index from a Row.
     * Shorthand: col(idx)
     * @param columnIndex The index of the column.
     * @return A function that extracts the column value from a Row.
     */
    public <T> Function<Row, T> col(int columnIndex) {
        return r -> r.get(columnIndex);
    }
}
