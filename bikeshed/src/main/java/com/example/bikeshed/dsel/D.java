package com.example.bikeshed.dsel;

import com.example.bikeshed.core.Series;

import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.function.Predicate;

/**
 * D.java is an omnibus enum for common DSEL operations,
 * acting as a namespace for "extension functions" and glyph-based shorthands.
 * This pattern centralizes functionality and improves discoverability.
 */
public enum D {
    // No specific enum instances are needed if all methods are static.
    // However, defining a dummy instance can prevent "no enum constants" error
    // and sometimes allows for certain meta-programming patterns (though not used here directly).
    OPS;

    /**
     * Glyph: `jn`
     * Shorthand for `Join.of(f, s)`. Creates an immutable 2-tuple.
     *
     * @param f   The first element.
     * @param s   The second element.
     * @param <F> Type of the first element.
     * @param <S> Type of the second element.
     * @return A new immutable Join instance.
     */
    public static <F, S> Join<F, S> jn(F f, S s) {
        return Join.of(f, s);
    }

    /**
     * Glyph: `sr`
     * Shorthand for `Series.of(size, provider)`. Creates a cursor-based collection.
     *
     * @param size     The number of elements in the series.
     * @param provider A function that provides an element given its index.
     * @param <T>      The type of elements.
     * @return A new immutable Series instance.
     */
    public static <T> Series<T> sr(int size, IntFunction<T> provider) {
        return Series.of(size, provider);
    }

    /**
     * Glyph: `mapFst`
     * Maps the first element of a Join to a new value.
     * Compositional: returns a new Join instance.
     *
     * @param join   The original Join instance.
     * @param mapper Function to apply to the first element.
     * @param <F>    Original type of the first element.
     * @param <S>    Type of the second element.
     * @param <R>    New type of the first element.
     * @return A new Join instance with the transformed first element.
     */
    public static <F, S, R> Join<R, S> mapFst(Join<F, S> join, Function<? super F, ? extends R> mapper) {
        return join.mapFst(mapper);
    }

    /**
     * Glyph: `mapSnd`
     * Maps the second element of a Join to a new value.
     * Compositional: returns a new Join instance.
     *
     * @param join   The original Join instance.
     * @param mapper Function to apply to the second element.
     * @param <F>    Type of the first element.
     * @param <S>    Original type of the second element.
     * @param <R>    New type of the second element.
     * @return A new Join instance with the transformed second element.
     */
    public static <F, S, R> Join<F, R> mapSnd(Join<F, S> join, Function<? super S, ? extends R> mapper) {
        return join.mapSnd(mapper);
    }

    /**
     * Glyph: `mapBoth`
     * Maps both elements of a Join to new values.
     * Compositional: returns a new Join instance.
     *
     * @param join       The original Join instance.
     * @param mapperFst  Function to apply to the first element.
     * @param mapperSnd  Function to apply to the second element.
     * @param <F>        Original type of the first element.
     * @param <S>        Original type of the second element.
     * @param <R1>       New type of the first element.
     * @param <R2>       New type of the second element.
     * @return A new Join instance with both elements transformed.
     */
    public static <F, S, R1, R2> Join<R1, R2> mapBoth(Join<F, S> join,
                                                     Function<? super F, ? extends R1> mapperFst,
                                                     Function<? super S, ? extends R2> mapperSnd) {
        return join.mapBoth(mapperFst, mapperSnd);
    }

    /**
     * Glyph: `swap`
     * Swaps the elements of a Join.
     * Compositional: returns a new Join instance.
     *
     * @param join The original Join instance.
     * @param <F>  Type of the first element.
     * @param <S>  Type of the second element.
     * @return A new Join instance with elements swapped.
     */
    public static <F, S> Join<S, F> swap(Join<F, S> join) {
        return join.swap();
    }

    /**
     * Glyph: `α` (alpha) - for map/transform operations.
     * Maps each element of a Series to a new type.
     * Compositional: returns a new Series instance.
     *
     * @param series The original Series.
     * @param mapper Function to apply to each element.
     * @param <T>    Original type of elements.
     * @param <R>    New type of elements.
     * @return A new Series with transformed elements.
     */
    public static <T, R> Series<R> alpha(Series<T> series, Function<? super T, ? extends R> mapper) {
        return series.map(mapper);
    }

    /**
     * Glyph: `filter`
     * Filters elements of a Series based on a predicate.
     * Compositional: returns a new Series instance (may be smaller).
     *
     * @param series    The original Series.
     * @param predicate Predicate to test each element.
     * @param <T>       Type of elements.
     * @return A new Series containing only elements that satisfy the predicate.
     */
    public static <T> Series<T> filter(Series<T> series, Predicate<? super T> predicate) {
        return series.filter(predicate);
    }

    /**
     * Glyph: `reduce`
     * Reduces a Series to a single value using a combining function.
     *
     * @param series      The original Series.
     * @param identity    The initial value of the accumulation.
     * @param accumulator A function that combines an accumulated result and an element.
     * @param <T>         Type of elements in the Series.
     * @param <U>         Type of the accumulated result.
     * @return The result of the reduction.
     */
    public static <T, U> U reduce(Series<T> series, U identity, java.util.function.BiFunction<U, ? super T, U> accumulator) {
        return series.reduce(identity, accumulator);
    }

    /**
     * Glyph: `toList`
     * Converts a Series to a standard Java List.
     *
     * @param series The Series to convert.
     * @param <T>    Type of elements.
     * @return A List containing all elements of the Series.
     */
    public static <T> java.util.List<T> toList(Series<T> series) {
        return series.toList();
    }

    /**
     * Glyph: `▶` (play button / iterator)
     * Provides an Iterable view of a Series.
     *
     * @param series The Series to view as Iterable.
     * @param <T>    Type of elements.
     * @return An Iterable that can be used in enhanced for loops.
     */
    public static <T> Iterable<T> iterable(Series<T> series) {
        return series; // Series implements Iterable
    }

    // Example of an "operator overloading via convention" for Series element access.
    // In actual Java, this would be `series.get(index)` or `series.at(index)`.
    // The "glyph" here is conceptual for the DSEL.
    // To enable `series[index]` like syntax in Java, one needs external libraries
    // or a compiler plugin (e.g., Manifold for Java, which allows extension methods).
    // For pure Java DSEL, we stick to method calls.
    public static <T> T get(Series<T> series, int index) {
        return series.get(index);
    }

    // Example of a "slice" operation that would ideally be `series[startIndex..endIndex]`
    public static <T> Series<T> slice(Series<T> series, int startIndex, int endIndex) {
        return series.slice(startIndex, endIndex);
    }
}
