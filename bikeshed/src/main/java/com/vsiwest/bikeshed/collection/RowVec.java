package com.vsiwest.bikeshed.collection;

import com.vsiwest.bikeshed.dsel.D;
import com.vsiwest.bikeshed.tuple.Join;
import com.vsiwest.bikeshed.types.ColumnMeta;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.IntFunction;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Represents a single row of data, where each element is a {@link Join} of a value and its {@link ColumnMeta}.
 * It extends {@link Series} to leverage its functional collection capabilities.
 * The second element of the Join is a {@link Supplier<ColumnMeta>} to allow for lazy metadata lookup,
 * which can be useful in scenarios where the schema is large or dynamically determined.
 */
public class RowVec extends Series<Join<Object, Supplier<ColumnMeta>>> {

    // Private constructor to enforce factory method usage
    private RowVec(int size, @NotNull IntFunction<Join<Object, Supplier<ColumnMeta>>> provider) {
        super(size, provider);
    }

    /**
     * Factory method to create a RowVec from a list of value-metadata pairs.
     *
     * @param valuesAndMeta A list of Join instances, where each Join contains the column value
     *                      and a Supplier for its ColumnMeta.
     * @return A new RowVec instance.
     */
    public static @NotNull RowVec of(@NotNull List<Join<Object, Supplier<ColumnMeta>>> valuesAndMeta) {
        Objects.requireNonNull(valuesAndMeta, "Values and meta list cannot be null.");
        return new RowVec(valuesAndMeta.size(), valuesAndMeta::get);
    }

    /**
     * Factory method to create a RowVec with a specified size and a provider function.
     *
     * @param size The number of columns in the row.
     * @param provider A function that provides a Join<Object, Supplier<ColumnMeta>> given its index.
     * @return A new RowVec instance.
     */
    public static @NotNull RowVec of(int size, @NotNull IntFunction<Join<Object, Supplier<ColumnMeta>>> provider) {
        return new RowVec(size, provider);
    }

    /**
     * Returns the value of a column at the specified index.
     *
     * @param colIndex The index of the column.
     * @return The value of the column.
     * @throws IndexOutOfBoundsException if the index is out of range.
     */
    public @Nullable Object getValue(int colIndex) {
        return get(colIndex).first();
    }

    /**
     * Returns the ColumnMeta of a column at the specified index.
     *
     * @param colIndex The index of the column.
     * @return The ColumnMeta of the column.
     * @throws IndexOutOfBoundsException if the index is out of range.
     */
    public @NotNull ColumnMeta getColumnMeta(int colIndex) {
        return get(colIndex).second().get();
    }

    /**
     * Returns the value of a column by its name.
     * This method iterates through the column metadata to find the matching name,
     * so it can be less efficient for very wide rows.
     *
     * @param columnName The name of the column.
     * @return The value of the column, or null if not found.
     */
    public @Nullable Object getValue(@NotNull String columnName) {
        return IntStream.range(0, size())
                .filter(i -> getColumnMeta(i).name().equals(columnName))
                .mapToObj(this::getValue)
                .findFirst()
                .orElse(null);
    }

    /**
     * Returns the ColumnMeta of a column by its name.
     *
     * @param columnName The name of the column.
     * @return The ColumnMeta of the column, or null if not found.
     */
    public @Nullable ColumnMeta getColumnMeta(@NotNull String columnName) {
        return IntStream.range(0, size())
                .filter(i -> getColumnMeta(i).name().equals(columnName))
                .mapToObj(this::getColumnMeta)
                .findFirst()
                .orElse(null);
    }

    /**
     * Returns a new RowVec containing only the specified columns by index.
     *
     * @param columnIndices The indices of the columns to select.
     * @return A new RowVec with the selected columns.
     */
    public @NotNull RowVec selectColumns(int... columnIndices) {
        Objects.requireNonNull(columnIndices, "Column indices array cannot be null.");
        return RowVec.of(columnIndices.length, i -> {
            int originalColIndex = columnIndices[i];
            return get(originalColIndex); // Return the original Join<Object, Supplier<ColumnMeta>>
        });
    }

    /**
     * Returns a new RowVec containing only the specified columns by name.
     *
     * @param columnNames The names of the columns to select.
     * @return A new RowVec with the selected columns.
     */
    public @NotNull RowVec selectColumnsByName(@NotNull String... columnNames) {
        Objects.requireNonNull(columnNames, "Column names array cannot be null.");
        // Build a map from column name to its original index for efficient lookup
        Map<String, Integer> nameToIndexMap = IntStream.range(0, size())
                .boxed()
                .collect(Collectors.toMap(i -> getColumnMeta(i).name(), i -> i));

        return RowVec.of(columnNames.length, i -> {
            String colName = columnNames[i];
            Integer originalColIndex = nameToIndexMap.get(colName);
            if (originalColIndex == null) {
                throw new IllegalArgumentException("Column '" + colName + "' not found in RowVec.");
            }
            return get(originalColIndex);
        });
    }

    @Override
    public String toString() {
        return "RowVec{" +
               IntStream.range(0, size())
                       .mapToObj(i -> {
                           ColumnMeta meta = getColumnMeta(i);
                           Object value = getValue(i);
                           return meta.name() + "=" + value + ":" + meta.type().name();
                       })
                       .collect(Collectors.joining(", ")) +
               '}';
    }
}
