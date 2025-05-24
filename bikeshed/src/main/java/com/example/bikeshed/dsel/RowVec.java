package com.example.bikeshed.dsel;

import com.example.bikeshed.core.Series;
import com.example.bikeshed.types.TypeMemento; // Assuming TypeMemento is the general interface
import com.example.bikeshed.types.ColumnMeta;

import java.util.function.IntFunction;
import java.util.function.Function;

/**
 * `RowVec` represents a row of data with associated metadata.
 * It is conceptually a `Series<Join<Any?, () -> ColumnMeta>>`.
 * Each element is a `Join` where the first part is the value and the second is a function
 * that provides `ColumnMeta` (lazily evaluated metadata).
 */
public class RowVec extends Series<Join<Object, Function<Void, ColumnMeta>>> {

    // Constructor to align with Series<T> constructor.
    // T is Join<Object, Function<Void, ColumnMeta>>
    protected RowVec(int size, IntFunction<Join<Object, Function<Void, ColumnMeta>>> provider) {
        super(size, provider);
    }

    /**
     * Factory method for `RowVec`.
     *
     * @param size The number of columns in the row.
     * @param provider A function providing the value-metadata pair for each column.
     * @return A new RowVec instance.
     */
    public static RowVec of(int size, IntFunction<Join<Object, Function<Void, ColumnMeta>>> provider) {
        return new RowVec(size, provider);
    }

    /**
     * Provides access to the value at a specific column index.
     * @param columnIndex The index of the column.
     * @return The value at that column.
     */
    public Object getValue(int columnIndex) {
        return this.get(columnIndex).getFirst();
    }

    /**
     * Provides access to the ColumnMeta at a specific column index.
     * @param columnIndex The index of the column.
     * @return The ColumnMeta for that column.
     */
    public ColumnMeta getColumnMeta(int columnIndex) {
        return this.get(columnIndex).getSecond().apply(null); // Invoke the supplier
    }

    /**
     * Gets the name of a column by index.
     * @param columnIndex The index of the column.
     * @return The name of the column.
     */
    public String getColumnName(int columnIndex) {
        return getColumnMeta(columnIndex).getName();
    }

    /**
     * Gets the type of a column by index.
     * @param columnIndex The index of the column.
     * @return The TypeMemento of the column.
     */
    public TypeMemento getColumnType(int columnIndex) {
        return getColumnMeta(columnIndex).getType();
    }
}
