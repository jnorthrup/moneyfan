package com.yourdomain.bikeshed.core;

import com.yourdomain.bikeshed.type.ColumnMeta;
import org.jetbrains.annotations.NotNull;

import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Represents a collection of {@link RowVec}s, providing a columnar abstraction
 * for tabular data. This is the primary interface for tabular data in the DSEL.
 *
 * In Kotlin: `typealias Cursor = Series<RowVec>`
 */
public interface Cursor extends Series<RowVec> {

    /**
     * Factory method to create a Cursor from a list of RowVecs.
     * @param rows A list of RowVec instances.
     * @return A new Cursor instance.
     */
    static @NotNull Cursor of(@NotNull List<RowVec> rows) {
        return Series.of(rows.size(), rows::get);
    }

    /**
     * Retrieves the {@link ColumnMeta} for each column in the first row.
     * Assumes all rows have the same schema.
     * @return A Series of ColumnMeta representing the schema.
     * @throws IllegalStateException if the cursor is empty.
     */
    default @NotNull Series<ColumnMeta> meta() {
        if (size() == 0) {
            throw new IllegalStateException("Cannot get meta from an empty Cursor.");
        }
        RowVec firstRow = get(0);
        return firstRow.alpha(columnValueAndMetaSupplier -> columnValueAndMetaSupplier.snd().get());
    }

    /**
     * Returns a new Cursor with only the specified columns.
     * This operation is compositional.
     * @param columnIndices An array of indices of the columns to retain.
     * @return A new Cursor with the selected columns.
     */
    default @NotNull Cursor selectColumns(@NotNull int... columnIndices) {
        // Create a new Series that provides RowVecs where each RowVec contains only the selected columns.
        return Cursor.of(size(), rowIndex -> {
            RowVec originalRow = get(rowIndex);
            // Create a new RowVec for the selected columns
            List<Join<Object, Supplier<ColumnMeta>>> selectedColumns = IntStream.of(columnIndices)
                    .mapToObj(originalRow::get)
                    .collect(Collectors.toList());
            return RowVec.of(selectedColumns);
        });
    }

    /**
     * Returns a new Cursor with columns filtered by names.
     * This involves resolving column names to indices via metadata.
     * @param columnNames Names of the columns to retain.
     * @return A new Cursor with the selected columns.
     */
    default @NotNull Cursor selectColumnsByName(@NotNull String... columnNames) {
        if (size() == 0) return Cursor.of(List.of()); // Return empty cursor if original is empty

        Series<ColumnMeta> currentMeta = meta();
        int[] columnIndices = new int[columnNames.length];
        for (int i = 0; i < columnNames.length; i++) {
            String targetName = columnNames[i];
            boolean found = false;
            for (int j = 0; j < currentMeta.size(); j++) {
                if (currentMeta.get(j).name().equals(targetName)) {
                    columnIndices[i] = j;
                    found = true;
                    break;
                }
            }
            if (!found) {
                throw new IllegalArgumentException("Column '" + targetName + "' not found in Cursor.");
            }
        }
        return selectColumns(columnIndices);
    }

    /**
     * Applies a function to each {@link RowVec} in the Cursor, producing a new Cursor with transformed rows.
     * @param rowMapper The function to apply to each RowVec.
     * @param <R> The type of the new RowVecs.
     * @return A new Cursor with transformed rows.
     */
    default <R extends RowVec> @NotNull Cursor mapRows(@NotNull Function<RowVec, R> rowMapper) {
        return this.alpha(rowMapper);
    }

    /**
     * Filters {@link RowVec}s in the Cursor based on a predicate, producing a new Cursor.
     * @param rowPredicate The predicate to filter RowVecs.
     * @return A new Cursor containing only rows that satisfy the predicate.
     */
    default @NotNull Cursor filterRows(@NotNull java.util.function.Predicate<RowVec> rowPredicate) {
        return this.filter(rowPredicate);
    }
}
