package com.moneyfan.dsel.core;

import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Provides static factory methods for creating instances of DSEL conceptual types
 * (like Series, RowVec, Cursor, ColumnMeta, Twin) which are all based on {@link Join}.
 * Also includes helper accessor methods for these structures.
 * <p>
 * The use of "conceptual types" means that types like {@code Series<T>} are actually
 * represented in Java code as {@code Join<Integer, Function<Integer, T>>}.
 * These factory methods provide semantic clarity and conciseness.
 */
public final class Types {

    private Types() {
        // Prevent instantiation of this utility class
    }

    // --- Join ---
    /** Short factory for Join. Same as {@link Join#jn(Object, Object)}. */
    public static <F, S> Join<F, S> jn(F first, S second) {
        return Join.jn(first, second);
    }

    // --- Type Aliases (Static Factory Methods for Join-based Structures) ---

    /** ColumnMeta = Join&lt;String, TypeMemento&gt; */
    public static Join<String, TypeMemento> cm(String name, TypeMemento type) {
        return jn(name, type);
    }

    /** Series&lt;T&gt; = Join&lt;Integer, Function&lt;Integer, T&gt;&gt; */
    public static <T> Join<Integer, Function<Integer, T>> sr(int size, Function<Integer, T> generator) {
        if (size < 0) throw new IllegalArgumentException("Series size cannot be negative.");
        return jn(size, generator);
    }

    /** RowVec = Series&lt;Join&lt;Object, Supplier&lt;ColumnMeta&gt;&gt;&gt; */
    public static Join<Integer, Function<Integer, Join<Object, Supplier<Join<String, TypeMemento>>>>> rv(
            int size,
            Function<Integer, Join<Object, Supplier<Join<String, TypeMemento>>>> rowGenerator
    ) {
        // RowVec is Series<CellType>, where CellType is Join<Object, Supplier<ColumnMeta>>
        // ColumnMeta is Join<String, TypeMemento>
        return sr(size, rowGenerator);
    }

    /** Cursor = Series&lt;RowVec&gt; */
    public static Join<Integer, Function<Integer, Join<Integer, Function<Integer, Join<Object, Supplier<Join<String, TypeMemento>>>>>>> cr(
            int size,
            Function<Integer, Join<Integer, Function<Integer, Join<Object, Supplier<Join<String, TypeMemento>>>>>> cursorRowGenerator
    ) {
        // Cursor is Series<RowVecType>
        return sr(size, cursorRowGenerator);
    }

    /** Twin&lt;T&gt; = Join&lt;T, T&gt; */
    public static <T> Join<T, T> tw(T first, T second) {
        return jn(first, second);
    }

    /** LongSeries&lt;T&gt; = Join&lt;Long, Function&lt;Long, T&gt;&gt; */
    public static <T> Join<Long, Function<Long, T>> lsr(long size, Function<Long, T> generator) {
        if (size < 0) throw new IllegalArgumentException("LongSeries size cannot be negative.");
        return jn(size, generator);
    }

    // --- Accessor Helpers for Join-based Structures ---

    // For Series<T> = Join<Integer, Function<Integer, T>>
    public static <T> int size(Join<Integer, Function<Integer, T>> series) {
        return series.f();
    }

    public static <T> T get(Join<Integer, Function<Integer, T>> series, int index) {
        if (index < 0 || index >= series.f()) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + series.f());
        }
        return series.s().apply(index);
    }

    // For LongSeries<T> = Join<Long, Function<Long, T>>
    public static <T> long lsize(Join<Long, Function<Long, T>> longSeries) {
        return longSeries.f();
    }

    public static <T> T lget(Join<Long, Function<Long, T>> longSeries, long index) {
        if (index < 0L || index >= longSeries.f()) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + longSeries.f());
        }
        return longSeries.s().apply(index);
    }

}
