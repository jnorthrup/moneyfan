package com.vsiwest.bikeshed.collection;

import com.vsiwest.bikeshed.dsel.D;
import org.jetbrains.annotations.NotNull;

import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Represents a collection of {@link RowVec} instances, acting as a table or dataset.
 * It extends {@link Series} where each element is a {@link RowVec}.
 * Provides DSEL-style operations for data manipulation.
 */
public class Cursor extends Series<RowVec> {

    // Private constructor to enforce factory method usage
    private Cursor(int size, @NotNull IntFunction<RowVec> provider) {
        super(size, provider);
    }

    /**
     * Factory method for `Cursor` from a list of `RowVec`s.
     *
     * @param rows The list of RowVecs.
     * @return A new Cursor instance.
     */
    public static @NotNull Cursor of(@NotNull List<RowVec> rows) {
        Objects.requireNonNull(rows, "Rows list cannot be null.");
        return new Cursor(rows.size(), rows::get);
    }

    /**
     * Factory method for `Cursor` with a specified size and a provider function.
     *
     * @param size The number of rows in the cursor.
     * @param provider A function that provides a RowVec given its index.
     * @return A new Cursor instance.
     */
    public static @NotNull Cursor of(int size, @NotNull IntFunction<RowVec> provider) {
        return new Cursor(size, provider);
    }

    /**
     * Returns the RowVec at the specified row index.
     *
     * @param rowIndex The index of the row.
     * @return The RowVec at the specified index.
     * @throws IndexOutOfBoundsException if the index is out of range.
     */
    public @NotNull RowVec row(int rowIndex) {
        return get(rowIndex);
    }

    /**
     * Slices the Cursor by row indices.
     *
     * @param range The range of rows to slice.
     * @return A new Cursor representing the sliced rows.
     */
    @Override
    public @NotNull Cursor slice(@NotNull IntRange range) {
        Series<RowVec> slicedSeries = super.slice(range);
        return new Cursor(slicedSeries.size(), slicedSeries::get);
    }

    /**
     * Selects columns from the Cursor by their indices.
     * Each row in the new Cursor will contain only the specified columns.
     *
     * @param columnIndices The indices of the columns to select.
     * @return A new Cursor with only the selected columns.
     */
    public @NotNull Cursor selectColumns(int... columnIndices) {
        Objects.requireNonNull(columnIndices, "Column indices array cannot be null.");
        return Cursor.of(this.size(), rowIndex -> this.get(rowIndex).selectColumns(columnIndices));
    }

    /**
     * Selects columns from the Cursor by their names.
     * Each row in the new Cursor will contain only the specified columns.
     *
     * @param columnNames The names of the columns to select.
     * @return A new Cursor with only the selected columns.
     */
    public @NotNull Cursor selectColumnsByName(@NotNull String... columnNames) {
        Objects.requireNonNull(columnNames, "Column names array cannot be null.");
        return Cursor.of(this.size(), rowIndex -> this.get(rowIndex).selectColumnsByName(columnNames));
    }

    /**
     * Applies a function to each {@link RowVec} in the Cursor, producing a new Cursor with transformed rows.
     *
     * @param rowMapper Function to apply to each RowVec.
     * @param <R> The type of the transformed RowVec (must extend RowVec).
     * @return A new Cursor with mapped rows.
     */
    public <R extends RowVec> @NotNull Cursor mapRows(@NotNull Function<RowVec, R> rowMapper) {
        Objects.requireNonNull(rowMapper, "Row mapper function cannot be null.");
        return Cursor.of(this.size(), rowIndex -> rowMapper.apply(this.get(rowIndex)));
    }

    /**
     * Filters {@link RowVec}s in the Cursor based on a predicate, producing a new Cursor.
     *
     * @param rowPredicate The predicate to filter RowVecs.
     * @return A new Cursor containing only rows that satisfy the predicate.
     */
    public @NotNull Cursor filterRows(@NotNull Predicate<RowVec> rowPredicate) {
        Objects.requireNonNull(rowPredicate, "Row predicate cannot be null.");
        List<RowVec> filteredRows = IntStream.range(0, size())
                .mapToObj(this::get)
                .filter(rowPredicate)
                .collect(Collectors.toList());
        return Cursor.of(filteredRows);
    }

    @Override
    public String toString() {
        return "Cursor(" + size() + " rows)";
    }
}
