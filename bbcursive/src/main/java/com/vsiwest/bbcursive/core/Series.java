package com.vsiwest.bbcursive.core;

import org.jetbrains.annotations.NotNull;
import java.util.function.IntFunction;
import java.util.function.Function;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.NoSuchElementException;
import java.util.Iterator;
import java.util.Objects;

public interface Series<T> extends Join<Integer, IntFunction<T>>, Iterable<T> {
    default int size() { return Objects.requireNonNull(first()); }
    default T get(int index) { if (index < 0 || index >= size()) throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size()); return Objects.requireNonNull(second()).apply(index); }
    default <R> Series<R> map(@NotNull Function<T, R> mapper) { return SeriesImpl.of(size(), i -> mapper.apply(get(i))); }
    default Series<T> filter(@NotNull java.util.function.Predicate<T> predicate) { List<T> l = this.toList().stream().filter(predicate).collect(Collectors.toList()); return SeriesImpl.of(l.size(), l::get); }
    default List<T> toList() { return IntStream.range(0, size()).mapToObj(this::get).collect(Collectors.toUnmodifiableList()); }
    @NotNull default Iterator<T> iterator() { return new Iterator<>() { private int current = 0; @Override public boolean hasNext() { return current < size(); } @Override public T next() { if (!hasNext()) throw new NoSuchElementException(); return get(current++); } }; }
    default boolean isEmpty() { return size() == 0; }

    final class SeriesImpl<T> extends Join<Integer, IntFunction<T>> implements Series<T> {
        SeriesImpl(Integer size, IntFunction<T> provider) { super(size, provider); }
        public static <T> Series<T> of(Integer size, IntFunction<T> provider) { return new SeriesImpl<>(size, provider); }
    }
}
