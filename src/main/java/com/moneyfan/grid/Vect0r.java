package com.moneyfan.grid;

import java.util.List;
import java.util.Objects;
import java.util.function.IntFunction;
import java.util.stream.IntStream;

/**
 * Lazy, immutable one-dimensional vector of {@code T} elements addressed by integer index.
 * <p>
 *     Internally the vector is represented by an {@code int size} and an {@link IntFunction}
 *     accessor that computes (or simply returns) the element at a given index.  No storage is
 *     allocated by this class; concrete backings (arrays, lists, mmap slices) can be wrapped by
 *     providing an accessor that captures the underlying container.
 * </p>
 */
public record Vect0r<T>(int size, IntFunction<? extends T> accessor) {

    public Vect0r {
        if (size < 0) throw new IllegalArgumentException("size must be >= 0");
        Objects.requireNonNull(accessor, "accessor");
    }

    /** Returns the element at {@code index}. */
    public T get(int index) {
        if (index < 0 || index >= size) throw new IndexOutOfBoundsException(index);
        return accessor.apply(index);
    }

    /** Simple mapping that creates another lazily-evaluated vector. */
    public <R> Vect0r<R> map(Class<R> newType, java.util.function.Function<? super T, ? extends R> mapper) {
        Objects.requireNonNull(mapper, "mapper");
        return new Vect0r<>(size, i -> mapper.apply(get(i)));
    }

    /** Iterates all elements eagerly applying consumer (useful for debugging/unit tests). */
    public void forEach(java.util.function.Consumer<? super T> consumer) {
        for (int i = 0; i < size; i++) consumer.accept(get(i));
    }

    // ===== factories =========================================================

    @SafeVarargs
    public static <T> Vect0r<T> of(T... values) {
        Objects.requireNonNull(values, "values");
        return new Vect0r<>(values.length, i -> values[i]);
    }

    public static <T> Vect0r<T> fromList(List<T> list) {
        Objects.requireNonNull(list, "list");
        return new Vect0r<>(list.size(), list::get);
    }

    /**
     * Builds a <em>materialised</em> list of all elements.  Primarily for testing and bridging to
     * traditional APIs; avoid in performance-sensitive paths.
     */
    public List<T> toList() {
        return IntStream.range(0, size).mapToObj(this::get).toList();
    }
}