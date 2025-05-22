package com.moneyfan.grid;

import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.function.IntFunction;

/**
 * Lazy, immutable vector backed by size and accessor function.
 * @param <T> element type
 */
public record Vect0r<T>(int size, IntFunction<T> accessor) implements Iterable<T> {

    public Vect0r {
        if (size < 0) throw new IllegalArgumentException("size must be >= 0");
        Objects.requireNonNull(accessor, "accessor");
    }

    public static <T> Vect0r<T> of(int size, IntFunction<T> accessor) {
        return new Vect0r<>(size, accessor);
    }

    public static <T> Vect0r<T> fromList(List<T> list) {
        Objects.requireNonNull(list, "list");
        return new Vect0r<>(list.size(), list::get);
    }

    public T get(int index) {
        if (index < 0 || index >= size) throw new IndexOutOfBoundsException(index);
        return accessor.apply(index);
    }

    @Override
    public Iterator<T> iterator() {
        return new Iterator<>() {
            private int idx = 0;

            @Override
            public boolean hasNext() {
                return idx < size;
            }

            @Override
            public T next() {
                return get(idx++);
            }
        };
    }
}