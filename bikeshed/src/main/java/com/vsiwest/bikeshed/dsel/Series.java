package com.vsiwest.bikeshed.dsel;

import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.function.Predicate;

/**
 * DSEL-specific `Series` providing additional convenience methods and adhering
 * to the `bbcursive` DSEL's philosophy.
 * It is effectively an alias for {@link com.example.bikeshed.core.Series}.
 */
public class Series<T> extends com.example.bikeshed.core.Series<T> {

    protected Series(Integer size, IntFunction<T> provider) {
        super(size, provider);
    }

    // DSEL-specific factory method, delegating to core.Series.of
    public static <T> Series<T> of(int size, IntFunction<T> provider) {
        return new Series<>(size, provider);
    }

    /**
     * Glyph for map/transform (`α`).
     * Maps each element of the Series to a new type.
     * Compositional: returns a new Series instance.
     *
     * @param mapper Function to apply to each element.
     * @param <R>    New type of elements.
     * @return A new Series with transformed elements.
     */
    public <R> Series<R> α(Function<? super T, ? extends R> mapper) {
        return map(mapper);
    }

    /**
     * Glyph for filter (`filter`).
     * Filters elements of the Series based on a predicate.
     * Compositional: returns a new Series instance (may be smaller).
     *
     * @param predicate Predicate to test each element.
     * @return A new Series containing only elements that satisfy the predicate.
     */
    public Series<T> filter(Predicate<? super T> predicate) {
        return super.filter(predicate);
    }

    /**
     * Operator-like method: `plus`. Concatenates two Series.
     *
     * @param other The other Series to concatenate.
     * @return A new Series containing elements from both.
     */
    public Series<T> plus(Series<T> other) {
        int newSize = this.size() + other.size();
        return D.sr(newSize, i -> {
            if (i < this.size()) {
                return this.get(i);
            } else {
                return other.get(i - this.size());
            }
        });
    }

    /**
     * Operator-like method: `get(index)`.
     * Direct element access by index.
     *
     * @param index The index.
     * @return The element at the index.
     */
    public T get(int index) {
        return super.get(index);
    }

    /**
     * Operator-like method: `get(range)`.
     * Slices the Series. For simplicity in Java, we'll use an explicit `IntRange` or similar.
     * For now, delegating to `slice(int, int)`.
     *
     * @param startIndex Inclusive start index.
     * @param endIndex Exclusive end index.
     * @return A new Series representing the slice.
     */
    public Series<T> get(int startIndex, int endIndex) {
        return super.slice(startIndex, endIndex);
    }

    /**
     * Glyph: `▶` (iterator)
     * Provides an Iterable view of the Series.
     *
     * @return An Iterable for this Series.
     */
    public Iterable<T> iteratorView() {
        return this; // This class already implements Iterable
    }
}
