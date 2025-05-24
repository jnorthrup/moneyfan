package com.vsiwest.moneyfan.bikeshed.core;

import org.jetbrains.annotations.NotNull;

import java.util.Iterator;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.IntFunction;

/**
 * Represents an immutable, lazily-evaluated sequence of elements.
 * This is a functional data structure, where operations return new Series instances
 * without modifying the original. It's optimized for cursor-based access patterns.
 *
 * @param <T> The type of elements in the series.
 */
public interface Series<T> extends Join<Integer, IntFunction<T>>, Iterable<T> {

    /**
     * Factory method to create a new Series instance.
     *
     * @param size The number of elements in the series.
     * @param provider A function that provides an element given its index.
     * @param <T> The type of elements.
     * @return A new Series instance.
     */
    static <T> Series<T> of(int size, IntFunction<T> provider) {
        Objects.requireNonNull(provider, "provider must not be null");
        if (size < 0) {
            throw new IllegalArgumentException("Size cannot be negative: " + size);
        }
        return new SeriesImpl<>(size, provider);
    }

    /**
     * Returns the number of elements in this series.
     *
     * @return The size of the series.
     */
    default int size() {
        return first();
    }

    /**
     * Returns the provider function for this series.
     *
     * @return The provider function.
     */
    default IntFunction<T> provider() {
        return second();
    }

    /**
     * Gets the element at the specified index.
     *
     * @param index The index of the element to retrieve.
     * @return The element at the given index.
     * @throws IndexOutOfBoundsException if the index is out of bounds.
     */
    default T get(int index) {
        if (index < 0 || index >= size()) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for Series of size " + size());
        }
        return provider().apply(index);
    }

    /**
     * Creates a new Series by applying a function to each element of this series.
     *
     * @param mapper The function to apply to each element.
     * @param <R> The type of elements in the new series.
     * @return A new Series with transformed elements.
     */
    default <R> Series<R> map(@NotNull Function<? super T, ? extends R> mapper) {
        Objects.requireNonNull(mapper, "mapper must not be null");
        return Series.of(size(), index -> mapper.apply(get(index)));
    }

    /**
     * Returns a new Series containing elements from this series within the specified range.
     * The new series is a view, not a copy of elements.
     *
     * @param range The inclusive range of indices to slice.
     * @return A new Series representing the slice.
     * @throws IndexOutOfBoundsException if the range is invalid (e.g., start < 0 or end > size).
     */
    default @NotNull Series<T> slice(@NotNull IntRange range) {
        int start = Math.max(0, range.from);
        int end = Math.min(size(), range.to + 1); // +1 because IntRange is inclusive of `to`
        int newSize = Math.max(0, end - start);

        return Series.of(newSize, index -> get(start + index));
    }

    /**
     * Returns a new Series containing elements from this series within the specified range.
     * The new series is a view, not a copy of elements.
     *
     * @param startIndex The inclusive starting index.
     * @param endIndex The exclusive ending index.
     * @return A new Series representing the slice.
     * @throws IndexOutOfBoundsException if the range is invalid.
     */
    default @NotNull Series<T> slice(int startIndex, int endIndex) {
        if (startIndex < 0 || endIndex > size() || startIndex > endIndex) {
            throw new IndexOutOfBoundsException("Invalid slice range: [" + startIndex + ", " + endIndex + ") for Series of size " + size());
        }
        int newSize = endIndex - startIndex;
        return Series.of(newSize, index -> get(startIndex + index));
    }

    /**
     * Returns an iterator over the elements in this series.
     *
     * @return An iterator.
     */
    @Override
    default @NotNull Iterator<T> iterator() {
        return new Iterator<>() {
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
     * Private inner class implementing the Series interface.
     * This is where the actual data (size and provider) is stored.
     */
    record SeriesImpl<T>(Integer first, IntFunction<T> second) implements Series<T> {
        // The record automatically provides constructor, accessors (first(), second()),
        // equals(), hashCode(), and toString().
        // No additional implementation needed here as the default methods in the interface
        // delegate to first() and second().
    }

    /**
     * Represents an inclusive integer range.
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
                // Or throw IllegalArgumentException if strictly valid ranges are required.
            }
        }
    }
}
