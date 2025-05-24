package com.vsiwest.moneyfan.bikeshed.dsel;

import com.vsiwest.moneyfan.bikeshed.core.Series;
import com.vsiwest.moneyfan.bikeshed.types.ColumnMeta;
import com.vsiwest.moneyfan.bikeshed.core.Join;

import java.util.function.IntFunction;
import java.util.List;
import java.util.stream.Collectors;
import java.util.function.Predicate;
import java.util.stream.IntStream;

public class Cursor extends Series<RowVec> {

    // Constructor to align with Series<T> where T is RowVec
    protected Cursor(int size, IntFunction<RowVec> provider) {
        super(size, provider);
    }

    /**
     * Factory method for `Cursor`.
     *
     * @param size The number of rows in the cursor.
     * @param provider A function that provides a RowVec given its row index.
     * @return A new Cursor instance.
     */
    public static Cursor of(int size, IntFunction<RowVec> provider) {
        return new Cursor(size, provider);
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
     * Gets the RowVec at the specified row index.
     *
     * @param rowIndex The index of the row to retrieve.
     * @return The RowVec at the given index.
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
     * Returns a new Cursor representing a slice of rows from this Cursor.
     *
     * @param startIndex The inclusive starting row index.
     * @param endIndex The exclusive ending row index.
     * @return A new Cursor representing the row slice.
     */
    public Cursor get(int startIndex, int endIndex) {
        // Delegating to the DSEL-specific Series.of for Cursor creation
        return D.sr(endIndex - startIndex, i -> get(startIndex + i));
    }

    /**
     * Returns a new Cursor representing a projection of columns from this Cursor.
     * This operation creates a new RowVec for each row, containing only the specified columns.
     *
     * @param columnIndices An array of column indices to include in the new Cursor.
     * @return A new Cursor with the projected columns.
     */
    public Cursor get(int... columnIndices) {
        // This is a row-major approach to columnar selection.
        // For actual columnar performance, data would be stored columnar.
        // This assumes RowVec can handle projection.
        return D.sr(this.size(), rowIndex -> {
            RowVec originalRow = this.get(rowIndex);
            return RowVec.of(columnIndices.length, colIndex -> {
                int originalColIndex = columnIndices[colIndex];
                // Assuming originalRow.get(originalColIndex) returns a Join<Object, Function<Void, ColumnMeta>>
                return originalRow.get(originalColIndex); // Get the original Join<Object, () -> ColumnMeta>
            });
        });
    }

    /**
     * Gets the metadata for all columns (from the first row, assuming homogeneous metadata).
     * @return A Series of ColumnMeta for all columns.
     */
    public Series<ColumnMeta> getMetaData() {
        if (this.size() == 0) {
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
        List<RowVec> filteredRows = IntStream.range(0, size())
                .mapToObj(this::get)
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
        List<R> mappedRows = IntStream.range(0, size())
                .mapToObj(this::get)
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
