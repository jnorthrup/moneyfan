package com.moneyfan.dsel;

import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.function.BiFunction;
import java.util.stream.Stream;

/**
 * Represents a sequence of elements, akin to a single column or
 * a sequence of records in pandas. Operations are generally lazy
 * or produce new Series instances, promoting immutability.
 *
 * @param <T> The type of elements in the series.
 */
public interface Series<T> extends Iterable<T> {

    <R> Series<R> map(Function<T, R> mapper);

    Series<T> filter(Predicate<T> predicate);

    <K> Series<Join<K, Series<T>>> groupBy(Function<T, K> classifier);

    // zip requires careful handling of series lengths
    <U, R> Series<R> zip(Series<U> other, BiFunction<T, U, R> zipper); // Changed Function<Join<T,U>,R> to BiFunction for directness

    <U, K, R> Series<R> join(
            Series<U> other,
            Function<T, K> thisKeyExtractor,
            Function<U, K> otherKeyExtractor,
            BiFunction<T, U, R> joinFunction,
            JoinType joinType
    );

    Series<T> sort(Comparator<? super T> comparator);

    <R> Series<R> rolling(int windowSize, Function<Series<T>, R> windowOperation);

    Series<T> shift(int periods, T fillValue); // Positive shifts forward, negative backward

    T reduce(T identity, BiFunction<T, T, T> accumulator);

    Optional<T> reduce(BiFunction<T, T, T> accumulator); // For non-empty series, no identity

    Series<T> distinct();

    long count();

    List<T> toList(); // Eagerly collects to a list

    Stream<T> stream();  // Updated to return Stream<T> for generic compatibility
}
