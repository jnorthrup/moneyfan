package com.moneyfan.grid;

import java.util.List;
import java.util.Objects;
import java.util.function.IntFunction;

/**
 * Lazy, immutable vector abstraction backed by an accessor function.
 *
 * @param <T> element type
 */
public record Vect0r<T>(int size, IntFunction<T> accessor) {

    public Vect0r {
        Objects.requireNonNull(accessor, "accessor");
        if (size < 0) {
            throw new IllegalArgumentException("size must be >= 0");
        }
    }

    /**
     * Factory creating a vector given size and accessor.
     */
    public static <T> Vect0r<T> of(int size, IntFunction<T> accessor) {
        return new Vect0r<>(size, accessor);
    }

    /**
     * Factory creating a vector backed by a list (eager backing).
     */
    public static <T> Vect0r<T> fromList(List<T> list) {
        Objects.requireNonNull(list, "list");
        return new Vect0r<>(list.size(), list::get);
    }

    /**
     * Returns the element at the provided index with bounds checking.
     */
    public T get(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("index=" + index);
        }
        return accessor.apply(index);
    }
}