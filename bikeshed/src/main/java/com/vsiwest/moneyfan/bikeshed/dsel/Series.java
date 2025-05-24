package com.vsiwest.moneyfan.bikeshed.dsel;

import com.vsiwest.moneyfan.bikeshed.core.Series;

import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.List;

/**
 * DSEL-specific extension of the core Series interface.
 * This class provides a concrete implementation of the Series interface
 * and can be used as the return type for DSEL operations that produce Series.
 * It delegates most of its functionality to the core Series implementation.
 *
 * @param <T> The type of elements in the series.
 */
public class Series<T> extends com.vsiwest.moneyfan.bikeshed.core.Series.SeriesImpl<T> {

    protected Series(Integer size, IntFunction<T> provider) {
        super(size, provider);
    }

    // DSEL-specific factory method, delegating to core.Series.of
    public static <T> Series<T> of(int size, IntFunction<T> provider) {
        return new Series<>(size, provider);
    }

    @Override
    public T get(int index) {
        return super.get(index);
    }

    /**
     * Returns a new DSEL Series representing a slice of elements from this Series.
     * This method overrides the default slice in core.Series to return a DSEL.Series.
     *
     * @param startIndex The inclusive starting index.
     * @param endIndex The exclusive ending index.
     * @return A new DSEL Series representing the slice.
     */
    public Series<T> get(int startIndex, int endIndex) {
        // Delegate to the core slice method, then wrap the result in a DSEL.Series
        return Series.of(endIndex - startIndex, i -> super.get(startIndex + i));
    }

    /**
     * Applies a function to each element of the Series, producing a new Series with transformed elements.
     * This is an "alpha conversion" or "map" operation, emphasizing compositional purity.
     *
     * @param mapper The function to apply to each element.
     * @param <R> The type of the new elements.
     * @return A new Series with transformed elements.
     */
    public <R> Series<R> alpha(Function<? super T, ? extends R> mapper) {
        return map(mapper); // Delegate to the core map method
    }

    /**
     * Filters elements of the Series based on a predicate, producing a new Series.
     * Note: This operation might not be strictly "cursor-based" in memory terms
     * if the underlying elements are not contiguous after filtering. For true
     * cursor semantics on filtered data, a new backing index would be required.
     * For simplicity, this returns a materialized list wrapped as a Series.
     *
     * @param predicate The predicate to filter elements.
     * @return A new Series containing only elements that satisfy the predicate.
     */
    public Series<T> filter(Predicate<? super T> predicate) {
        List<T> filteredList = IntStream.range(0, size())
                .mapToObj(this::get)
                .filter(predicate)
                .collect(Collectors.toList());
        return Series.of(filteredList.size(), filteredList::get);
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
     * Provides an Iterable view of the Series.
     *
     * @return An Iterable for this Series.
     */
    public Iterable<T> iteratorView() {
        return this; // This class already implements Iterable
    }
}
