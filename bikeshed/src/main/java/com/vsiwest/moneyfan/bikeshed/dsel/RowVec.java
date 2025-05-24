package com.vsiwest.moneyfan.bikeshed.dsel;

import com.vsiwest.moneyfan.bikeshed.core.Join;
import com.vsiwest.moneyfan.bikeshed.core.Series;
import com.vsiwest.moneyfan.bikeshed.types.ColumnMeta;

import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.List;

public class RowVec extends Series<Join<Object, Function<Void, ColumnMeta>>> {

    // Constructor to align with Series<T> constructor.
    // T is Join<Object, Function<Void, ColumnMeta>>
    protected RowVec(int size, IntFunction<Join<Object, Function<Void, ColumnMeta>>> provider) {
        super(size, provider);
    }

    /**
     * Factory method for `RowVec`.
     *
     * @param size The number of columns in the row vector.
     * @param provider A function that provides a column's value and its metadata supplier given its column index.
     * @return A new RowVec instance.
     */
    public static RowVec of(int size, IntFunction<Join<Object, Function<Void, ColumnMeta>>> provider) {
        return new RowVec(size, provider);
    }

    /**
     * Factory method for `RowVec` from a List of value-metadata pairs.
     *
     * @param valuesAndMeta A list of Join<Object, Function<Void, ColumnMeta>> representing the columns.
     * @return A new RowVec instance.
     */
    public static RowVec of(List<Join<Object, Function<Void, ColumnMeta>>> valuesAndMeta) {
        return new RowVec(valuesAndMeta.size(), valuesAndMeta::get);
    }

    /**
     * Provides access to the value at a specific column index.
     * @param columnIndex The index of the column.
     * @return The value at that column.
     */
    public Object getValue(int columnIndex) {
        return this.get(columnIndex).first();
    }

    /**
     * Gets the ColumnMeta for a specific column index.
     * This assumes the second element of the Join is a supplier for ColumnMeta.
     *
     * @param columnIndex The index of the column.
     * @return The ColumnMeta for the specified column.
     */
    public ColumnMeta getColumnMeta(int columnIndex) {
        // The 'get' method from Series returns Join<Object, Function<Void, ColumnMeta>>
        // We then get the second element (the Function<Void, ColumnMeta>) and apply it.
        return this.get(columnIndex).second().apply(null); // Invoke the supplier
    }

    /**
     * Gets the name of a column by index.
     * @param columnIndex The index of the column.
     * @return The name of the column.
     */
    public String getColumnName(int columnIndex) {
        return getColumnMeta(columnIndex).name();
    }

    /**
     * Gets the type of a column by index.
     * @param columnIndex The index of the column.
     * @return The TypeMemento of the column.
     */
    public com.vsiwest.moneyfan.bikeshed.types.TypeMemento getColumnType(int columnIndex) {
        return getColumnMeta(columnIndex).type();
    }
}
