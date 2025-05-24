package com.vsiwest.moneyfan.bikeshed.dsel;

import com.vsiwest.moneyfan.bikeshed.core.Join;
import com.vsiwest.moneyfan.bikeshed.core.Series;
import com.vsiwest.moneyfan.bikeshed.types.ColumnMeta;
import com.vsiwest.moneyfan.bikeshed.types.IOMemento; // Assuming IOMemento is in this package
import com.vsiwest.moneyfan.bikeshed.types.TypeMemento;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.function.Supplier;

/**
 * D - The Domain-Specific Embedded Language (DSEL) for data processing.
 * This enum provides static-like methods that act as global DSEL operations,
 * offering a concise, "glyph-based shorthand" style for common data manipulations.
 * It emphasizes immutability, compositional purity, and function references.
 */
public enum D {
    // This enum doesn't need instances; it's a namespace for utility methods.
    ; // No enum constants

    public static final String META_SFX = ".meta";

    // --- Join Operations (Glyphs: jn, fst, snd, mapBoth, swap, test) ---

    /**
     * Shorthand for {@code Join.of(f, s)}. Creates an immutable 2-tuple.
     * Glyph: `jn`
     * Example: `D.jn("hello", 123)`
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
     * Returns the first element of a Join.
     * Glyph: `f` (read as "first")
     * Example: `D.f(D.jn("abc", 123))`
     *
     * @param join The Join instance.
     * @param <F> Type of the first element.
     * @param <S> Type of the second element.
     * @return The first element.
     */
    public static <F, S> F f(@NotNull Join<F, S> join) {
        return join.first();
    }

    /**
     * Returns the second element of a Join.
     * Glyph: `s` (read as "second")
     * Example: `D.s(D.jn("abc", 123))`
     *
     * @param join The Join instance.
     * @param <F> Type of the first element.
     * @param <S> Type of the second element.
     * @return The second element.
     */
    public static <F, S> S s(@NotNull Join<F, S> join) {
        return join.second();
    }

    /**
     * Tests a Join against a predicate.
     * Glyph: `test`
     * Example: `D.test(D.jn("hello", 123), j -> D.f(j).startsWith("h") && D.s(j) > 100)`
     *
     * @param join The Join instance to test.
     * @param predicate The predicate to apply.
     * @param <F> Type of the first element.
     * @param <S> Type of the second element.
     * @return True if the Join satisfies the predicate, false otherwise.
     */
    public static <F, S> boolean test(@NotNull Join<F, S> join, @NotNull Predicate<Join<F, S>> predicate) {
        return predicate.test(join);
    }

    // --- Series Operations (Glyphs: sr, size, get, alpha, filter, iter, toList, first, last, head, tail, skip, each) ---

    /**
     * Shorthand for {@code Series.of(size, provider)}. Creates a new Series.
     * Glyph: `sr`
     * Example: `D.sr(10, i -> "Item " + i)`
     *
     * @param size The number of elements.
     * @param provider A function to provide elements by index.
     * @param <T> Type of elements.
     * @return A new Series instance.
     */
    public static <T> @NotNull Series<T> sr(int size, @NotNull Function<Integer, T> provider) {
        return Series.of(size, provider);
    }

    /**
     * Returns the size of a Series. Shorthand for {@code series.size()}.
     * Glyph: `size` or `sz`
     * Example: `D.size(mySeries)`
     *
     * @param series The input Series.
     * @param <T> Element type.
     * @return The size of the Series.
     */
    public static <T> int size(@NotNull Series<T> series) {
        return series.size();
    }

    /**
     * Returns the element at a specific index in a Series. Shorthand for {@code series.get(index)}.
     * Glyph: `get`
     * Example: `D.get(mySeries, 5)`
     *
     * @param series The input Series.
     * @param index The index.
     * @param <T> Element type.
     * @return The element at the specified index.
     */
    public static <T> T get(@NotNull Series<T> series, int index) {
        return series.get(index);
    }

    /**
     * Applies a function to each element of a Series (alpha conversion/map).
     * Glyph: `alpha`
     * Example: `D.alpha(mySeries, String::toUpperCase)`
     *
     * @param series The input Series.
     * @param mapper The transformation function.
     * @param <T> Original element type.
     * @param <R> New element type.
     * @return A new Series with transformed elements.
     */
    public static <T, R> @NotNull Series<R> alpha(@NotNull Series<T> series, @NotNull Function<T, R> mapper) {
        return series.alpha(mapper);
    }

    /**
     * Filters a Series based on a predicate.
     * Glyph: `filter` or `fil`
     * Example: `D.filter(mySeries, s -> s.length() > 5)`
     *
     * @param series The input Series.
     * @param predicate The predicate to filter elements.
     * @param <T> Element type.
     * @return A new Series containing filtered elements.
     */
    public static <T> @NotNull Series<T> filter(@NotNull Series<T> series, @NotNull Predicate<T> predicate) {
        return series.filter(predicate);
    }

    /**
     * Provides an Iterable view of a Series.
     * Glyph: `iter`
     * Example: `for (String s : D.iter(mySeries)) { ... }`
     *
     * @param series The input Series.
     * @param <T> Element type.
     * @return An Iterable view of the Series.
     */
    public static <T> @NotNull Iterable<T> iter(@NotNull Series<T> series) {
        return series; // Series already implements Iterable
    }

    /**
     * Converts a Series to a List. Shorthand for {@code series.toList()}.
     * Glyph: `toList` or `ls`
     * Example: `D.toList(mySeries)`
     *
     * @param series The input Series.
     * @param <T> Element type.
     * @return A new List containing all elements of the Series.
     */
    public static <T> @NotNull List<T> toList(@NotNull Series<T> series) {
        return series.toList();
    }

    /**
     * Returns the first element of a Series. Shorthand for {@code series.first()}.
     * Glyph: `first` or `fst`
     * Example: `D.first(mySeries)`
     *
     * @param series The input Series.
     * @param <T> Element type.
     * @return The first element.
     */
    public static <T> T first(@NotNull Series<T> series) {
        return series.first();
    }

    /**
     * Returns the last element of a Series. Shorthand for {@code series.last()}.
     * Glyph: `last` or `lst`
     * Example: `D.last(mySeries)`
     *
     * @param series The input Series.
     * @param <T> Element type.
     * @return The last element.
     */
    public static <T> T last(@NotNull Series<T> series) {
        return series.last();
    }

    /**
     * Returns a new Series containing elements from the beginning up to (but not including) the specified end index.
     * Shorthand for {@code series.head(exclusiveEnd)}.
     * Glyph: `head` or `hd`
     * Example: `D.head(mySeries, 5)`
     *
     * @param series The input Series.
     * @param exclusiveEnd The exclusive end index.
     * @param <T> Element type.
     * @return A new Series representing the head.
     */
    public static <T> @NotNull Series<T> head(@NotNull Series<T> series, int exclusiveEnd) {
        return series.head(exclusiveEnd);
    }

    /**
     * Returns a new Series containing elements from the specified start index to the end.
     * Shorthand for {@code series.tail(inclusiveStart)}.
     * Glyph: `tail` or `tl`
     * Example: `D.tail(mySeries, 5)`
     *
     * @param series The input Series.
     * @param inclusiveStart The inclusive start index.
     * @param <T> Element type.
     * @return A new Series representing the tail.
     */
    public static <T> @NotNull Series<T> tail(@NotNull Series<T> series, int inclusiveStart) {
        return series.tail(inclusiveStart);
    }

    /**
     * Returns a new Series skipping the first 'n' elements. Shorthand for {@code series.skip(n)}.
     * Glyph: `skip` or `sk`
     * Example: `D.skip(mySeries, 2)`
     *
     * @param series The input Series.
     * @param n The number of elements to skip.
     * @param <T> Element type.
     * @return A new Series with elements skipped.
     */
    public static <T> @NotNull Series<T> skip(@NotNull Series<T> series, int n) {
        return series.skip(n);
    }

    /**
     * Executes an action for each element in the Series. Shorthand for {@code series.each(action)}.
     * Glyph: `each`
     * Example: `D.each(mySeries, System.out::println)`
     *
     * @param series The input Series.
     * @param action The action to perform.
     * @param <T> Element type.
     */
    public static <T> void each(@NotNull Series<T> series, @NotNull java.util.function.Consumer<T> action) {
        series.each(action);
    }

    // --- Reactive Series Operations (combine, scan) ---

    /**
     * Combines two Series into a Series of Joins (pairs).
     * The resulting Series will have a length equal to the minimum of the lengths of the input Series.
     * Each Join will contain one element from {@code seriesF} and one from {@code seriesS} at the corresponding index.
     * Glyph: `combine`
     * Example: `D.combine(D.sr(3, i -> "A"+i), D.sr(2, i -> i*10))` would result in a `Series<Join<String, Integer>>` of size 2: `[("A0", 0), ("A1", 10)]`.
     *
     * @param seriesF The first Series.
     * @param seriesS The second Series.
     * @param <F> Type of elements in the first Series.
     * @param <S> Type of elements in the second Series.
     * @return A new Series of Joins.
     */
    public static <F, S> @NotNull Series<Join<F, S>> combine(@NotNull Series<F> seriesF, @NotNull Series<S> seriesS) {
        int newSize = Math.min(seriesF.size(), seriesS.size());
        return Series.of(newSize, index -> Join.of(seriesF.get(index), seriesS.get(index)));
    }

    /**
     * Applies an accumulator function over a Series, producing a new Series of intermediate results.
     * The first element of the output Series is the {@code initialValue}.
     * Each subsequent element is the result of applying the accumulator function to the previously computed value
     * and the current element from the input Series. The output Series will have one more element than the input Series.
     * Glyph: `scan`
     * Example: `D.scan(D.sr(3, i -> i+1), 0, (sum, item) -> sum + item)` would result in `Series<Integer>`: `[0, 1, 3, 6]`. (Input Series: `[1, 2, 3]`)
     *
     * @param seriesT The input Series.
     * @param initialValue The initial value for the accumulation.
     * @param accumulator The function to apply to the current accumulation and the current item.
     * @param <T> Type of elements in the input Series.
     * @param <R> Type of elements in the output Series (and the accumulation).
     * @return A new Series of accumulated results.
     */
    public static <T, R> @NotNull Series<R> scan(@NotNull Series<T> seriesT, R initialValue, @NotNull BiFunction<R, T, R> accumulator) {
        List<R> results = new ArrayList<>();
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
     * Glyph: `rv`
     * Example: `D.rv(List.of(D.jn(10, () -> colMeta1), D.jn("text", () -> colMeta2)))`
     *
     * @param valuesAndMeta A list of value-metadata supplier pairs for the row.
     * @return A new RowVec instance.
     */
    public static @NotNull RowVec rv(@NotNull List<Join<Object, Supplier<ColumnMeta>>> valuesAndMeta) {
        return RowVec.of(valuesAndMeta);
    }

    /**
     * Returns the column name from a RowVec at a given index.
     * Glyph: `colName`
     * Example: `D.colName(myRowVec, 0)`
     *
     * @param rowVec The RowVec.
     * @param colIndex The column index.
     * @return The name of the column.
     */
    public static @NotNull String colName(@NotNull RowVec rowVec, int colIndex) {
        return rowVec.get(colIndex).second().get().name();
    }

    /**
     * Returns the TypeMemento of a column from a RowVec at a given index.
     * Glyph: `colType`
     * Example: `D.colType(myRowVec, 0)`
     *
     * @param rowVec The RowVec.
     * @param colIndex The column index.
     * @return The TypeMemento of the column.
     */
    public static @NotNull TypeMemento colType(@NotNull RowVec rowVec, int colIndex) {
        return rowVec.get(colIndex).second().get().type();
    }

    // --- Cursor Operations (Glyphs: cur, slc, scn, snm, mapRows, filterRows) ---

    /**
     * Shorthand for {@code Cursor.of(rows)}. Creates a new Cursor.
     * Glyph: `cur`
     * Example: `D.cur(List.of(rowVec1, rowVec2))`
     *
     * @param rows A list of RowVecs.
     * @return A new Cursor instance.
     */
    public static @NotNull Cursor cur(@NotNull List<RowVec> rows) {
        return Cursor.of(rows);
    }

    /**
     * Slices a Cursor by row indices. Shorthand for {@code cursor.slice(range)}.
     * Glyph: `slc`
     * Example: `D.slc(myCursor, Series.IntRange.of(0, 9))`
     *
     * @param cursor The input Cursor.
     * @param range The range of rows to slice.
     * @return A new Cursor representing the sliced rows.
     */
    public static @NotNull Cursor slc(@NotNull Cursor cursor, @NotNull Series.IntRange range) {
        return cursor.slice(range);
    }

    /**
     * Selects columns from a Cursor by index. Shorthand for {@code cursor.selectColumns(indices)}.
     * Glyph: `scn` (Select Columns by Number/Index)
     * Example: `D.scn(myCursor, 0, 2, 4)`
     *
     * @param cursor The input Cursor.
     * @param columnIndices Indices of columns to select.
     * @return A new Cursor with only the selected columns.
     */
    public static @NotNull Cursor scn(@NotNull Cursor cursor, int... columnIndices) {
        return cursor.selectColumns(columnIndices);
    }

    /**
     * Selects columns from a Cursor by name. Shorthand for {@code cursor.selectColumnsByName(names)}.
     * Glyph: `snm` (Select Columns by Name)
     * Example: `D.snm(myCursor, "Open", "Close")`
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
     * Shorthand for {@code cursor.mapRows(rowMapper)}.
     * Glyph: `mapRows`
     * Example: `D.mapRows(myCursor, row -> D.alpha(row, val -> val.toString().toUpperCase()))`
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
     * Shorthand for {@code cursor.filterRows(rowPredicate)}.
     * Glyph: `filterRows`
     * Example: `D.filterRows(myCursor, row -> (Long)D.get(row, 0).first() > 100L)`
     *
     * @param cursor The input Cursor.
     * @param rowPredicate The predicate to filter RowVecs.
     * @return A new Cursor containing only rows that satisfy the predicate.
     */
    public static @NotNull Cursor filterRows(@NotNull Cursor cursor, @NotNull Predicate<RowVec> rowPredicate) {
        return cursor.filterRows(rowPredicate);
    }

    // --- ColumnMeta Operations (Glyphs: cm) ---

    /**
     * Shorthand for {@code ColumnMeta.of(name, type)}. Creates a new ColumnMeta.
     * Glyph: `cm`
     * Example: `D.cm("Age", IOMemento.IoInt)`
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
     * Glyph: `fsString`
     * Example: `D.fsString(20)`
     *
     * @param length The fixed length of the string in bytes.
     * @return A TypeMemento representing a fixed-size string.
     */
    public static @NotNull TypeMemento fsString(int length) {
        return new FixedSizeTypeMemento(IOMemento.IoString, length);
    }

    /**
     * Creates a fixed-size binary blob TypeMemento for ISAM.
     * Glyph: `fsBinaryBlob`
     * Example: `D.fsBinaryBlob(100)`
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
