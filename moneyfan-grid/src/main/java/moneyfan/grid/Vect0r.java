package moneyfan.grid;

import java.util.List;
import java.util.Objects;
import java.util.function.IntFunction;

public record Vect0r<T>(int size, IntFunction<T> accessor) {
    public static <T> Vect0r<T> of(int size, IntFunction<T> accessor) {
        Objects.requireNonNull(accessor);
        if (size < 0) throw new IllegalArgumentException("size must be non-negative");
        return new Vect0r<>(size, accessor);
    }

    public static <T> Vect0r<T> fromList(List<T> list) {
        Objects.requireNonNull(list);
        return new Vect0r<>(list.size(), list::get);
    }

    public T get(int index) {
        if (index < 0 || index >= size) throw new IndexOutOfBoundsException(index);
        return accessor.apply(index);
    }
}