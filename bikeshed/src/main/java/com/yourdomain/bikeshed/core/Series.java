package com.yourdomain.bikeshed.core;

import org.jetbrains.annotations.NotNull;

import java.util.Iterator;
import java.util.List;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * A cursor-based collection, representing a sequence of elements of type T.
 * Implemented as a {@code Join<Integer, Function<Integer, T>>} where the first element
 * is the size and the second is a provider function (like a getter by index).
 * This enables lazy evaluation for element access.
 *
 * @param <T> The type of elements in the Series.
 */
public interface Series<T> extends Join<Integer, Function<Integer, T>>, Iterable<T> {

    /**
     * Factory method to create a new Series instance.
     *
     * @param size The number of elements in the series.
     * @param provider A function that provides an element given its index.
     * @param <T> The type of elements.
     * @return A new Series instance.
     */
    static <T> @NotNull Series<T> of(int size, @NotNull Function<Integer, T> provider) {
        return new ImmutableSeries<>(size, provider);
    }

    /**
     * Returns the number of elements in this Series.
     * @return The size of the series.
     */
    default int size() {
        return fst();
    }

    /**
     * Returns the element at the specified index.
     * @param index The index of the element.
     * @return The element at the given index.
     * @throws IndexOutOfBoundsException if the index is out of range (index < 0 || index >= size()).
     */
    default T get(int index) {
        if (index < 0 || index >= size()) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for Series of size " + size());
        }
        return snd().apply(index);
    }

    /**
     * Returns a new Series containing elements from this Series within the specified range.
     * This operation is compositional; it creates a new view rather than modifying the original.
     *
     * @param range The range of indices to include.
     * @return A new Series representing the slice.
     */
    default @NotNull Series<T> slice(@NotNull IntRange range) {
        int start = Math.max(0, range.from);
        int end = Math.min(size(), range.to + 1); // +1 because IntRange is inclusive of `to`
        int newSize = Math.max(0, end - start);

        return Series.of(newSize, index -> get(start + index));
    }

    /**
     * Applies a function to each element of the Series, producing a new Series with transformed elements.
     * This is an "alpha conversion" or "map" operation, emphasizing compositional purity.
     *
     * @param mapper The function to apply to each element.
     * @param <R> The type of the new elements.
     * @return A new Series with transformed elements.
     */
    default <R> @NotNull Series<R> alpha(@NotNull Function<T, R> mapper) {
        return Series.of(size(), index -> mapper.apply(get(index)));
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
    default @NotNull Series<T> filter(@NotNull Predicate<T> predicate) {
        List<T> filteredList = IntStream.range(0, size())
                .mapToObj(this::get)
                .filter(predicate)
                .collect(Collectors.toList());
        return Series.of(filteredList.size(), filteredList::get);
    }

    /**
     * Provides an iterator over the elements of this Series.
     * @return An iterator.
     */
    @NotNull
    @Override
    default Iterator<T> iterator() {
        return new Iterator<T>() {
            private int currentIndex = 0;
            @Override
            public boolean hasNext() {
                return currentIndex < size();
            }

            @Override
            public T next() {
                if (!hasNext()) {
                    throw new java.util.NoSuchElementException();
                }
                return get(currentIndex++);
            }
        };
    }

    /**
     * Converts the Series to a List.
     * @return A new List containing all elements of the Series.
     */
    default @NotNull List<T> toList() {
        return IntStream.range(0, size()).mapToObj(this::get).collect(Collectors.toList());
    }

    /**
     * Returns the first element of the Series.
     * @return The first element.
     * @throws java.util.NoSuchElementException if the Series is empty.
     */
    default T first() {
        if (size() == 0) throw new java.util.NoSuchElementException("Series is empty.");
        return get(0);
    }

    /**
     * Returns the last element of the Series.
     * @return The last element.
     * @throws java.util.NoSuchElementException if the Series is empty.
     */
    default T last() {
        if (size() == 0) throw new java.util.NoSuchElementException("Series is empty.");
        return get(size() - 1);
    }

    /**
     * Returns a new Series containing elements from the beginning up to (but not including) the specified end index.
     * @param exclusiveEnd The exclusive end index.
     * @return A new Series representing the head.
     */
    default @NotNull Series<T> head(int exclusiveEnd) {
        return slice(new IntRange(0, exclusiveEnd - 1));
    }

    /**
     * Returns a new Series containing elements from the specified start index to the end.
     * @param inclusiveStart The inclusive start index.
     * @return A new Series representing the tail.
     */
    default @NotNull Series<T> tail(int inclusiveStart) {
        return slice(new IntRange(inclusiveStart, size() - 1));
    }

    /**
     * Returns a new Series skipping the first 'n' elements.
     * @param n The number of elements to skip.
     * @return A new Series with elements skipped.
     */
    default @NotNull Series<T> skip(int n) {
        return slice(new IntRange(n, size() - 1));
    }

    /**
     * Executes an action for each element in the Series.
     * @param action The action to perform.
     */
    default void each(@NotNull java.util.function.Consumer<T> action) {
        for (T item : this) {
            action.accept(item);
        }
    }

    /**
     * Represents an inclusive range of integers, useful for slicing.
     * In Java, `IntStream.range` is exclusive of the end, so `to` needs `+1`.
     * This custom `IntRange` makes it inclusive of `to`.
     */
    final class IntRange {
        public final int from;
        public final int to; // Inclusive

        public IntRange(int from, int to) {
            this.from = from;
            this.to = to;
            if (from > to) {
                // Handle empty or invalid ranges gracefully, depending on desired semantics
                // For slicing, an empty range can be represented by from > to.
            }
        }

        public static @NotNull IntRange of(int from, int to) {
            return new IntRange(from, to);
        }

        @Override
        public String toString() {
            return "[" + from + ".." + to + "]";
        }
    }

    // Inner class for the immutable implementation
    final class ImmutableSeries<T> extends Join.ImmutableJoin<Integer, Function<Integer, T>> implements Series<T> {
        private ImmutableSeries(Integer size, Function<Integer, T> provider) {
            super(size, provider);
        }
    }
}
