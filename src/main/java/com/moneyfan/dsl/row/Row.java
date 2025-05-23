package com.moneyfan.dsl.row;

import com.moneyfan.dsl.typeevidence.SchemaEvidence;

/**
 * Represents a row of data. This is an abstract interface.
 * Implementations will define how data is stored and accessed.
 * Adheres to the "2-ary tuple abstract interface" concept by being composable,
 * where a Row could be conceptualized as Join<Schema, Values> or a sequence of Join<ColumnName, Value>.
 */
public interface Row {
    /**
     * Gets a value from the row by its column name.
     * @param columnName The name of the column.
     * @param <T> The expected type of the value.
     * @return The value.
     */
    <T> T get(String columnName);

    /**
     * Gets a value from the row by its column index.
     * @param columnIndex The index of the column.
     * @param <T> The expected type of the value.
     * @return The value.
     */
    <T> T get(int columnIndex);

    SchemaEvidence schema();
    Object[] getValues(); // For direct access to all values, e.g., for bulk operations
}
