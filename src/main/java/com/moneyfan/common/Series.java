package com.moneyfan.common;

import java.util.stream.Stream;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;

// A Series represents an ordered sequence of elements of type E.
// It extends Iterable to allow for-each looping and provides stream-based processing.
public interface Series<E> extends Iterable<E> {

    // Returns the number of elements in this series.
    long size();

    // Retrieves an element at the specified position in this series.
    E get(long index);

    // Returns a sequential Stream over the elements in this series.
    Stream<E> stream();

    // Returns a new Series consisting of the results of applying the given
    // function to the elements of this series. (Unary Operator)
    default <R> Series<R> map(Function<? super E, ? extends R> mapper) {
        // Default implementation collects to a list and wraps in a new ListSeries.
        // Concrete implementations might offer more efficient or lazy mapping.
        return new ListSeries<>(this.stream().map(mapper).collect(Collectors.toList()));
    }

    // Returns a new Series consisting of the elements of this series that match
    // the given predicate. (Unary Operator)
    default Series<E> filter(Predicate<? super E> predicate) {
        // Default implementation collects to a list and wraps in a new ListSeries.
        // Concrete implementations might offer more efficient or lazy filtering.
        return new ListSeries<>(this.stream().filter(predicate).collect(Collectors.toList()));
    }
    // Other common DSEL operations (e.g., take, drop, reduce, forEach) can be added.
}
