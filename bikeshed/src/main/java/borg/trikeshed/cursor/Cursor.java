package borg.trikeshed.cursor; // Corrected package

import borg.trikeshed.cursor.RowVec;
import borg.trikeshed.isam.RecordMeta;
import borg.trikeshed.lib.Join;
import borg.trikeshed.lib.Series;
import org.jetbrains.annotations.NotNull;

import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.Objects; // Added for requireNonNull

/**
 * Represents a collection of {@link RowVec}s, providing a columnar abstraction
 * for tabular data. This is the primary interface for tabular data in the DSEL.
 *
 * In Kotlin: `typealias Cursor = Series<RowVec>`
 */
public interface Cursor extends borg.trikeshed.lib.Series<borg.trikeshed.cursor.RowVec> {

    /**
     * Factory method to create a Cursor from a list of RowVecs.
     * @param rows A list of RowVec instances.
     * @return A new Cursor instance.
     * @deprecated prefer {@link #of(Series)}
     */
    @Deprecated
    static @NotNull Cursor of(@NotNull List<borg.trikeshed.cursor.RowVec> rows) {
        return borg.trikeshed.lib.Series.of(rows.size(), rows::get);
    }

    /**
     * Factory method to create a Cursor from a Series of RowVecs.
     * @param rowsSeries A Series of RowVec instances.
     * @return A new Cursor instance.
     */
    static @NotNull Cursor of(@NotNull Series<RowVec> rowsSeries) {
        return new DelegatingCursor(rowsSeries);
    }

    // Inner class to wrap a Series<RowVec> into a Cursor
    // This ensures that any Cursor is also a valid Series<RowVec> by delegation.
    static class DelegatingCursor implements Cursor { // Made static nested class
        private final Series<RowVec> underlyingSeries;

        private DelegatingCursor(Series<RowVec> series) {
            this.underlyingSeries = Objects.requireNonNull(series, "underlyingSeries must not be null");
        }

        // Delegate Series<RowVec> methods which form the basis of Cursor
        @Override
        public Integer fst() {
            return underlyingSeries.fst(); // This is the size from the underlying Series
        }

        @Override
        public Function<Integer, RowVec> snd() {
            return underlyingSeries.snd(); // This is the provider from the underlying Series
        }

        // Default methods from Series (like size(), get(), iterator()) will use fst() and snd().
        // Default methods from Cursor (like meta(), selectColumns()) will use get(), size() etc. from Series.
        // No need to override them unless specific behavior is required beyond delegation.
    }

    /**
     * Retrieves the {@link ColumnMeta} for each column in the first row.
     * Assumes all rows have the same schema.
     * @return A Series of ColumnMeta representing the schema.
     * @throws IllegalStateException if the cursor is empty.
     */
    default @NotNull borg.trikeshed.lib.Series<borg.trikeshed.isam.RecordMeta> meta() {
        if (size() == 0) {
            throw new IllegalStateException("Cannot get meta from an empty Cursor.");
        }
        borg.trikeshed.cursor.RowVec firstRow = get(0); // This get() is from Series<RowVec>
        // firstRow is a RowVec, which is a Series<Join<Object, Supplier<RecordMeta>>> (after RowVec is refactored)
        // The lambda's input `join` is an element of firstRow, so it's a Join<Object, Supplier<RecordMeta>>.
        return firstRow.alpha(join -> ((java.util.function.Supplier<borg.trikeshed.isam.RecordMeta>)join.snd()).get());
    }

    /**
     * Returns a new Cursor with only the specified columns.
     * This operation is compositional.
     * @param columnIndices An array of indices of the columns to retain.
     * @return A new Cursor with the selected columns.
     */
    default @NotNull Cursor selectColumns(@NotNull int... columnIndices) {
        // Create a new Series that provides RowVecs where each RowVec contains only the selected columns.
        // Use the new Cursor.of(Series<RowVec>) factory
        return Cursor.of(Series.of(size(), rowIndex -> {
            borg.trikeshed.cursor.RowVec originalRow = get(rowIndex); // get() from Series<borg.trikeshed.cursor.RowVec>

            // originalRow.get() returns Join<Object, Supplier<RecordMeta>>
            // Collect directly to an array of the correct Join type.
            // The cast inside mapToObj previously was problematic if originalRow.get() wasn't exactly that type.
            // Assuming originalRow.get() returns the correctly typed Join object.
            @SuppressWarnings("unchecked") // Suppress warning for creating generic array; RowVec.of expects Join[]
            borg.trikeshed.lib.Join<Object, java.util.function.Supplier<borg.trikeshed.isam.RecordMeta>>[] selectedColumnsArray =
                IntStream.of(columnIndices)
                    .mapToObj(originalRow::get) // This is IntFunction<Join<Object, Supplier<RecordMeta>>>
                    .toArray(borg.trikeshed.lib.Join[]::new); // Create an array of Join

            return borg.trikeshed.cursor.RowVec.of(selectedColumnsArray); // RowVec.of takes varargs (Join...)
        }));
    }

    /**
     * Returns a new Cursor with columns filtered by names.
     * This involves resolving column names to indices via metadata.
     * @param columnNames Names of the columns to retain.
     * @return A new Cursor with the selected columns.
     */
    default @NotNull Cursor selectColumnsByName(@NotNull String... columnNames) {
        if (size() == 0) return Cursor.of(java.util.List.of()); // Return empty cursor if original is empty

        borg.trikeshed.lib.Series<borg.trikeshed.isam.RecordMeta> currentMeta = meta();
        int[] columnIndices = new int[columnNames.length];
        for (int i = 0; i < columnNames.length; i++) {
            String targetName = columnNames[i];
            boolean found = false;
            for (int j = 0; j < currentMeta.size(); j++) {