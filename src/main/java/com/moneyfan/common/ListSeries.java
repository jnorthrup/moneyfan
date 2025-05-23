package com.moneyfan.common;

import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.function.Function;
import java.util.function.Predicate;

// A basic List-backed implementation of the Series interface.
// This class is package-private as users should interact via the Series interface.
class ListSeries<E> implements Series<E> {
    private final List<E> data;

    public ListSeries(List<E> data) {
        this.data = Objects.requireNonNull(data, "Data list cannot be null");
    }

    public static <T> Series<T> empty() {
        return new ListSeries<>(Collections.emptyList());
    }

    @Override
    public long size() {
        return data.size();
    }

    @Override
    public E get(long index) {
        if (index < 0 || index >= data.size()) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + data.size());
        }
        return data.get((int) index); // List.get expects int
    }

    @Override
    public Stream<E> stream() {
        return data.stream();
    }

    @Override
    public Iterator<E> iterator() {
        return data.iterator();
    }

    // Overrides for map and filter can be more efficient or specialized if needed,
    // but default implementations in Series interface (using stream().collect()) are sufficient for now.
    // For example, if lazy evaluation is desired, these would be implemented differently.
}
