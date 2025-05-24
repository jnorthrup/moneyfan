package com.vsiwest.moneyfan.bikeshed.dsel;

import com.vsiwest.moneyfan.bikeshed.core.Series;
import com.vsiwest.moneyfan.bikeshed.types.ColumnMeta;
import com.vsiwest.moneyfan.bikeshed.core.Join;

import java.util.function.IntFunction;
import java.util.List;
import java.util.stream.Collectors;
import java.util.function.Predicate;
import java.util.stream.IntStream;

/**
 * `Cursor` represents a collection of `RowVec`s, providing columnar abstraction.
 * This is the primary interface for tabular data within the DSEL.
 * It is conceptually a `Series<RowVec>`.
 */
public class Cursor extends Series<RowVec> {

    // Constructor to align with Series<T> where T is RowVec
    protected Cursor(int size, IntFunction<RowVec> provider) {
        super(size, provider);
    }

    /**
     * Factory method for `Cursor`.
     *
     * @param numRows The number of rows in the cursor.
     * @param rowProvider A function providing a RowVec for each row index.
     * @return A new Cursor instance.
     */
    public static Cursor of(int numRows, IntFunction<RowVec> rowProvider) {
        return new Cursor(numRows, rowProvider);
    }

    /**
     * Factory method for `Cursor` from a List of `RowVec`s.
     *
     * @param rows A list of RowVecs.
     * @return A new Cursor instance.
     */
    public static Cursor of(List<RowVec> rows) {
        return new Cursor(rows.size(), rows::get);
    }

    /**
     * Gets a row by its index.
     * Glyph: `at` or `row`.
     * @param rowIndex The index of the row.
     * @return The RowVec at the specified row index.
     */
    public RowVec at(int rowIndex) {
        return get(rowIndex);
    }

    /**
     * Gets a row by its index (alias for `at`).
     * @param rowIndex The index of the row.
     * @return The RowVec at the specified row index.
     */
    public RowVec row(int rowIndex) {
        return at(rowIndex);
    }

    /**
     * Gets a sub-Cursor containing a range of rows.
     * @param startIndex The inclusive start index of the rows.
     * @param endIndex The exclusive end index of the rows.
     * @return A new Cursor containing the specified range of rows.
     */
    public Cursor get(int startIndex, int endIndex) {
        return D.sr(endIndex - startIndex, i -> get(startIndex + i));
    }

    /**
     * Gets a sub-Cursor containing a selection of columns.
     * Note: This materializes selected columns for simplicity; a lazy columnar projection
     * would be more performant for very wide tables.
     * @param columnIndices An array of column indices to select.
     * @return A new Cursor with only the specified columns.
     */
    public Cursor get(int... columnIndices) {
        // This is a row-major approach to columnar selection.
        // For actual columnar performance, data would be stored columnar.
        // This assumes RowVec can handle projection.
        return D.sr(this.size(), rowIndex -> {
            RowVec originalRow = this.get(rowIndex);
            return RowVec.of(columnIndices.length, colIndex -> {
                int originalColIndex = columnIndices[colIndex];
                return originalRow.get(originalColIndex); // Get the original Join<Object, () -> ColumnMeta>
            });
        });
    }

    /**
     * Gets the metadata for all columns (from the first row, assuming homogeneous metadata).
     * @return A Series of ColumnMeta for all columns.
     */
    public Series<ColumnMeta> getMetaData() {
        if (this.isEmpty()) {
            return D.sr(0, i -> { throw new IllegalStateException("Cannot get metadata from empty cursor."); });
        }
        RowVec firstRow = this.get(0);
        return D.sr(firstRow.size(), i -> firstRow.getColumnMeta(i));
    }

    /**
     * Filters rows based on a predicate applied to each `RowVec`.
     * Compositional: returns a new Cursor (may be smaller).
     * @param predicate A predicate applied to each RowVec.
     * @return A new Cursor containing only rows that satisfy the predicate.
     */
    public Cursor filterRows(Predicate<RowVec> predicate) {
        List<RowVec> filteredRows = D.toList(this) // Materialize to filter
                .stream()
                .filter(predicate)
                .collect(Collectors.toList());
        return D.sr(filteredRows.size(), filteredRows::get);
    }

    /**
     * Applies a function to each {@link RowVec} in the Cursor, producing a new Cursor with transformed rows.
     * @param rowMapper Function to apply to each RowVec.
     * @param <R> The type of the transformed RowVec.
     * @return A new Cursor with mapped rows.
     */
    public <R extends RowVec> Cursor mapRows(Function<RowVec, R> rowMapper) {
        List<R> mappedRows = D.toList(this) // Materialize to map
                .stream()
                .map(rowMapper)
                .collect(Collectors.toList());
        return D.sr(mappedRows.size(), mappedRows::get);
    }

    /**
     * Selects columns from a Cursor by index.
     * @param columnIndices Indices of columns to select.
     * @return A new Cursor with only the selected columns.
     */
    public Cursor selectColumns(int... columnIndices) {
        return get(columnIndices);
    }

    /**
     * Selects columns from a Cursor by name.
     * This method first resolves column names to indices using the metadata from the first row.
     * @param columnNames Names of columns to select.
     * @return A new Cursor with only the selected columns.
     * @throws IllegalArgumentException if any specified column name is not found.
     */
    public Cursor selectColumnsByName(String... columnNames) {
        if (this.size() == 0) {
            throw new IllegalStateException("Cannot select columns by name from an empty cursor.");
        }
        Series<ColumnMeta> metadata = getMetaData();
        int[] indices = new int[columnNames.length];
        for (int i = 0; i < columnNames.length; i++) {
            String nameToFind = columnNames[i];
            int foundIndex = -1;
            for (int j = 0; j < metadata.size(); j++) {
                if (metadata.get(j).name().equals(nameToFind)) {
                    foundIndex = j;
                    break;
                }
            }
            if (foundIndex == -1) {
                throw new IllegalArgumentException("Column '" + nameToFind + "' not found in cursor metadata.");
            }
            indices[i] = foundIndex;
        }
        return get(indices);
    }
}
