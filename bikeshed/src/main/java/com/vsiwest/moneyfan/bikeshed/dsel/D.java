package com.vsiwest.moneyfan.bikeshed.dsel;

import com.vsiwest.moneyfan.bikeshed.core.Join;
import com.vsiwest.moneyfan.bikeshed.core.Series;
import com.vsiwest.moneyfan.bikeshed.types.ColumnMeta;
import com.vsiwest.moneyfan.bikeshed.types.IOMemento; // Assuming IOMemento is in this package
import com.vsiwest.moneyfan.bikeshed.types.TypeMemento;
import org.jetbrains.annotations.NotNull;

import java.util.List;
import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.function.Predicate;
import java.util.function.Supplier;

/**
 * DSEL (Domain-Specific Embedded Language) utility class.
 * This enum doesn't need instances; it's a namespace for utility methods,
 * acting as "extension functions" or "glyphs" for the DSEL.
 */
public enum D {
    ; // No enum constants

    public static final String META_SFX = ".meta";

    // --- Join Operations (Glyphs: jn, fst, snd, mapBoth, swap, test) ---

    /**
     * Shorthand for {@code Join.of(f, s)}. Creates an immutable 2-tuple.
     *
     * @param f The first element.
     * @param s The second element.
     * @param <F> Type of the first element.
     * @param <S> Type of the second element.
     * @return A new Join instance.
     */
    public static <F, S> @NotNull Join<F, S> jn(F f, S s) {
        return Join.of(f, s);
    }

    /**
     * Shorthand for {@code join.first()}.
     *
     * @param join The Join instance.
     * @param <F> Type of the first element.
     * @param <S> Type of the second element.
     * @return The first element.
     */
    public static <F, S> F fst(@NotNull Join<F, S> join) {
        Objects.requireNonNull(join, "Join must not be null");
        return join.first();
    }

    /**
     * Shorthand for {@code join.second()}.
     *
     * @param join The Join instance.
     * @param <F> Type of the first element.
     * @param <S> Type of the second element.
     * @return The second element.
     */
    public static <F, S> S snd(@NotNull Join<F, S> join) {
        Objects.requireNonNull(join, "Join must not be null");
        return join.second();
    }

    /**
     * Shorthand for {@code join.mapBoth(mapperFst, mapperSnd)}.
     *
     * @param join The Join instance.
     * @param mapperFst Function to apply to the first element.
     * @param mapperSnd Function to apply to the second element.
     * @param <F> Original type of the first element.
     * @param <S> Original type of the second element.
     * @param <R1> Result type of the first element.
     * @param <R2> Result type of the second element.
     * @return A new Join with transformed elements.
     */
    public static <F, S, R1, R2> @NotNull Join<R1, R2> mapBoth(@NotNull Join<F, S> join,
                                                               Function<? super F, ? extends R1> mapperFst,
                                                               Function<? super S, ? extends R2> mapperSnd) {
        Objects.requireNonNull(join, "Join must not be null");
        return join.mapBoth(mapperFst, mapperSnd);
    }

    /**
     * Shorthand for {@code join.swap()}.
     *
     * @param join The Join instance.
     * @param <F> Type of the first element.
     * @param <S> Type of the second element.
     * @return A new Join with elements swapped.
     */
    public static <F, S> @NotNull Join<S, F> swap(@NotNull Join<F, S> join) {
        Objects.requireNonNull(join, "Join must not be null");
        return join.swap();
    }

    // --- Series Operations (Glyphs: sr, size, get, map, slice) ---

    /**
     * Shorthand for {@code Series.of(size, provider)}. Creates a new Series.
     *
     * @param size The number of elements.
     * @param provider A function to provide elements by index.
     * @param <T> The type of elements.
     * @return A new Series instance.
     */
    public static <T> @NotNull Series<T> sr(int size, @NotNull IntFunction<T> provider) {
        return Series.of(size, provider);
    }

    /**
     * Shorthand for {@code series.size()}.
     *
     * @param series The Series instance.
     * @param <T> The type of elements.
     * @return The size of the series.
     */
    public static <T> int size(@NotNull Series<T> series) {
        Objects.requireNonNull(series, "Series must not be null");
        return series.size();
    }

    /**
     * Shorthand for {@code series.get(index)}.
     *
     * @param series The Series instance.
     * @param index The index.
     * @param <T> The type of elements.
     * @return The element at the given index.
     */
    public static <T> T get(@NotNull Series<T> series, int index) {
        Objects.requireNonNull(series, "Series must not be null");
        return series.get(index);
    }

    /**
     * Shorthand for {@code series.map(mapper)}.
     *
     * @param series The Series instance.
     * @param mapper The mapping function.
     * @param <T> Original type of elements.
     * @param <R> Result type of elements.
     * @return A new Series with transformed elements.
     */
    public static <T, R> @NotNull Series<R> map(@NotNull Series<T> series, @NotNull Function<? super T, ? extends R> mapper) {
        Objects.requireNonNull(series, "Series must not be null");
        return series.map(mapper);
    }

    /**
     * Shorthand for {@code series.slice(startIndex, endIndex)}.
     *
     * @param series The Series instance.
     * @param startIndex The inclusive start index.
     * @param endIndex The exclusive end index.
     * @param <T> The type of elements.
     * @return A new Series representing the slice.
     */
    public static <T> @NotNull Series<T> slice(@NotNull Series<T> series, int startIndex, int endIndex) {
        Objects.requireNonNull(series, "Series must not be null");
        return series.slice(startIndex, endIndex);
    }

    /**
     * Converts a Series to a List.
     * @param series The input Series.
     * @param <T> Element type.
     * @return A new List containing all elements of the Series.
     */
    public static <T> @NotNull List<T> toList(@NotNull Series<T> series) {
        return series.toList();
    }

    /**
     * Returns the first element of a Series.
     * @param series The input Series.
     * @param <T> Element type.
     * @return The first element.
     */
    public static <T> T first(@NotNull Series<T> series) {
        return series.first();
    }

    /**
     * Returns the last element of a Series.
     * @param series The input Series.
     * @param <T> Element type.
     * @return The last element.
     */
    public static <T> T last(@NotNull Series<T> series) {
        return series.last();
    }

    /**
     * Applies an accumulator function over a Series, producing a new Series of intermediate results.
     * The first element of the output Series is the {@code initialValue}.
     * Each subsequent element is the result of applying the accumulator function to the previously computed value
     * and the current element from the input Series. The output Series will have one more element than the input Series.
     *
     * @param seriesT The input Series.
     * @param initialValue The initial value for the accumulation.
     * @param accumulator The function to apply to the current accumulation and the current item.
     * @param <T> Type of elements in the input Series.
     * @param <R> Type of elements in the output Series (and the accumulation).
     * @return A new Series of accumulated results.
     */
    public static <T, R> @NotNull Series<R> scan(@NotNull Series<T> seriesT, R initialValue, @NotNull BiFunction<R, T, R> accumulator) {
        List<R> results = new java.util.ArrayList<>();
        results.add(initialValue);
        R currentAccumulation = initialValue;
        for (int i = 0; i < seriesT.size(); i++) {
            T currentItem = seriesT.get(i);
            currentAccumulation = accumulator.apply(currentAccumulation, currentItem);
            results.add(currentAccumulation);
        }
        return Series.of(results.size(), results::get);
    }

    // --- RowVec Operations (Glyphs: rv, colName, colType) ---

    /**
     * Shorthand for {@code RowVec.of(valuesAndMeta)}. Creates a new RowVec.
     *
     * @param valuesAndMeta A list of value-metadata supplier pairs for the row.
     * @return A new RowVec instance.
     */
    public static @NotNull RowVec rv(@NotNull List<Join<Object, Supplier<ColumnMeta>>> valuesAndMeta) {
        return RowVec.of(valuesAndMeta.size(), valuesAndMeta::get);
    }

    /**
     * Returns the column name from a RowVec at a given index.
     *
     * @param rowVec The RowVec.
     * @param colIndex The column index.
     * @return The name of the column.
     */
    public static @NotNull String colName(@NotNull RowVec rowVec, int colIndex) {
        return rowVec.getColumnMeta(colIndex).name();
    }

    /**
     * Returns the TypeMemento of a column from a RowVec at a given index.
     *
     * @param rowVec The RowVec.
     * @param colIndex The column index.
     * @return The TypeMemento of the column.
     */
    public static @NotNull TypeMemento colType(@NotNull RowVec rowVec, int colIndex) {
        return rowVec.getColumnMeta(colIndex).type();
    }

    // --- Cursor Operations (Glyphs: cur, slc, scn, snm, mapRows, filterRows) ---

    /**
     * Shorthand for {@code Cursor.of(rows)}. Creates a new Cursor.
     *
     * @param rows A list of RowVecs.
     * @return A new Cursor instance.
     */
    public static @NotNull Cursor cur(@NotNull List<RowVec> rows) {
        return Cursor.of(rows);
    }

    /**
     * Slices a Cursor by row indices.
     *
     * @param cursor The input Cursor.
     * @param range The range of rows to slice.
     * @return A new Cursor representing the sliced rows.
     */
    public static @NotNull Cursor slc(@NotNull Cursor cursor, @NotNull Series.IntRange range) {
        return cursor.slice(range);
    }

    /**
     * Selects columns from a Cursor by index.
     *
     * @param cursor The input Cursor.
     * @param columnIndices Indices of columns to select.
     * @return A new Cursor with only the selected columns.
     */
    public static @NotNull Cursor scn(@NotNull Cursor cursor, int... columnIndices) {
        return cursor.selectColumns(columnIndices);
    }

    /**
     * Selects columns from a Cursor by name.
     *
     * @param cursor The input Cursor.
     * @param columnNames Names of columns to select.
     * @return A new Cursor with only the selected columns.
     */
    public static @NotNull Cursor snm(@NotNull Cursor cursor, @NotNull String... columnNames) {
        return cursor.selectColumnsByName(columnNames);
    }

    /**
     * Applies a function to each {@link RowVec} in the Cursor, producing a new Cursor with transformed rows.
     *
     * @param cursor The input Cursor.
     * @param rowMapper Function to apply to each RowVec.
     * @param <R> The type of the transformed RowVec.
     * @return A new Cursor with mapped rows.
     */
    public static <R extends RowVec> @NotNull Cursor mapRows(@NotNull Cursor cursor, @NotNull Function<RowVec, R> rowMapper) {
        return cursor.mapRows(rowMapper);
    }

    /**
     * Filters {@link RowVec}s in the Cursor based on a predicate, producing a new Cursor.
     *
     * @param cursor The input Cursor.
     * @param rowPredicate The predicate to filter RowVecs.
     * @return A new Cursor containing only rows that satisfy the predicate.
     */
    public static <R extends RowVec> @NotNull Cursor filterRows(@NotNull Cursor cursor, @NotNull Predicate<RowVec> rowPredicate) {
        return cursor.filterRows(rowPredicate);
    }

    // --- ColumnMeta Operations (Glyphs: cm) ---

    /**
     * Shorthand for {@code ColumnMeta.of(name, type)}. Creates a new ColumnMeta.
     *
     * @param name The name of the column.
     * @param type The TypeMemento describing the column's data type.
     * @return A new ColumnMeta instance.
     */
    public static @NotNull ColumnMeta cm(@NotNull String name, @NotNull TypeMemento type) {
        return ColumnMeta.of(name, type);
    }

    // --- TypeMemento/IOMemento Fixed-Size String/Binary Blob Factories (Glyphs: fsString, fsBinaryBlob) ---

    /**
     * Creates a fixed-size string TypeMemento for ISAM.
     *
     * @param length The fixed length of the string in bytes.
     * @return A TypeMemento representing a fixed-size string.
     */
    public static @NotNull TypeMemento fsString(int length) {
        return new FixedSizeTypeMemento(IOMemento.IoString, length);
    }

    /**
     * Creates a fixed-size binary blob TypeMemento for ISAM.
     *
     * @param length The fixed length of the binary blob in bytes.
     * @return A TypeMemento representing a fixed-size binary blob.
     */
    public static @NotNull TypeMemento fsBinaryBlob(int length) {
        return new FixedSizeTypeMemento(IOMemento.IoByteArray, length);
    }

    /**
     * Internal helper class for fixed-size variable-length types (String, ByteArray)
     * when used in ISAM, where their length is determined by schema, not content.
     */
    public static class FixedSizeTypeMemento implements TypeMemento {
        private final IOMemento baseType;
        private final int fixedSize;

        public FixedSizeTypeMemento(@NotNull IOMemento baseType, int fixedSize) {
            if (baseType.networkSize() != null) {
                throw new IllegalArgumentException("FixedSizeTypeMemento is only for variable-length IOMementos (String, ByteArray).");
            }
            this.baseType = baseType;
            this.fixedSize = fixedSize;
        }

        @Override
        public Integer networkSize() {
            return fixedSize;
        }

        public IOMemento getBaseType() {
            return baseType;
        }

        @Override
        public String toString() {
            return baseType.name() + "(" + fixedSize + ")";
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            FixedSizeTypeMemento that = (FixedSizeTypeMemento) o;
            return fixedSize == that.fixedSize && baseType == that.baseType;
        }

        @Override
        public int hashCode() {
            return java.util.Objects.hash(baseType, fixedSize);
        }
    }
}
