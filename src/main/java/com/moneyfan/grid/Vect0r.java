package com.moneyfan.grid;

import java.util.List;
import java.util.function.IntFunction;

/**
 * Lazy, immutable vector record.
 */
public record Vect0r<T>(int size, IntFunction<T> accessor) {
    
    public static <T> Vect0r<T> of(int size, IntFunction<T> accessor) {
        return new Vect0r<>(size, accessor);
    }
    
    public static <T> Vect0r<T> fromList(List<T> list) {
        return new Vect0r<>(list.size(), list::get);
    }
    
    public static <T> Vect0r<T> empty() {
        return new Vect0r<>(0, i -> null);
    }
    
    public T get(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size);
        }
        return accessor.apply(index);
    }
    
    public <R> Vect0r<R> map(IntFunction<R> mapper) {
        return new Vect0r<>(size, mapper);
    }
    
    public <R> Vect0r<R> map(java.util.function.Function<T, R> mapper) {
        return new Vect0r<>(size, i -> mapper.apply(get(i)));
    }
}