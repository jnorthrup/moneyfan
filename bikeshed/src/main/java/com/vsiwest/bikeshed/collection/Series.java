package com.vsiwest.bikeshed.collection;

import com.vsiwest.bikeshed.tuple.Join;
import org.jetbrains.annotations.NotNull;

import java.util.AbstractList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Represents an immutable, lazily-evaluated sequence of elements.
 * It's a functional data structure, providing a view over data rather than owning it.
 * Implements {@link Join} to represent its size and a provider function.
 * Implements {@link Iterable} for easy iteration.
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
    static <T> @NotNull Series<T> of(int size, @NotNull IntFunction<T> provider) {
        if (size < 0) {
            throw new IllegalArgumentException("Size cannot be negative.");
        }
        return new ImmutableSeries<>(size, provider);
    }

    /**
     * Factory method to create a Series from a List.
     *
     * @param list The list of elements.
     * @param <T> The type of elements.
     * @return A new Series instance backed by the list.
     */
    static <T> @NotNull Series<T> of(@NotNull List<T> list) {
        Objects.requireNonNull(list, "List cannot be null.");
        return new ImmutableSeries<>(list.size(), list::get);
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
     * Returns the element at the specified position in this series.
     *
     * @param index The index of the element to return.
     * @return The element at the given index.
     * @throws IndexOutOfBoundsException if the index is out of range (index < 0 || index >= size()).
     */
    default T get(int index) {
        if (index < 0 || index >= size()) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size());
        }
        return second().apply(index);
    }

    /**
     * Applies a mapping function to each element of this series, producing a new series.
     * This is a lazy transformation.
     *
     * @param mapper The function to apply to each element.
     * @param <R> The type of elements in the new series.
     * @return A new Series with the transformed elements.
     */
    default <R> @NotNull Series<R> map(@NotNull Function<T, R> mapper) {
        Objects.requireNonNull(mapper, "Mapper function cannot be null.");
        return Series.of(size(), index -> mapper.apply(get(index)));
    }

    /**
     * Filters elements of this series based on a predicate, producing a new series.
     * Note: This operation might materialize the filtered elements into a list
     * to maintain efficient indexed access, depending on the underlying implementation.
     *
     * @param predicate The predicate to apply to each element.
     * @return A new Series containing only the elements that satisfy the predicate.
     */
    default @NotNull Series<T> filter(@NotNull Predicate<T> predicate) {
        Objects.requireNonNull(predicate, "Predicate cannot be null.");
        List<T> filteredList = IntStream.range(0, size())
                .mapToObj(this::get)
                .filter(predicate)
                .collect(Collectors.toList());
        return Series.of(filteredList);
    }

    /**
     * Returns a new Series representing a slice of this series.
     * The slice is inclusive of `range.from` and `range.to`.
     *
     * @param range The range of indices to slice.
     * @return A new Series representing the specified slice.
     */
    default @NotNull Series<T> slice(@NotNull IntRange range) {
        int start = Math.max(0, range.from);
        int end = Math.min(size(), range.to + 1); // +1 because IntRange is inclusive of `to`
        int newSize = Math.max(0, end - start);

        return Series.of(newSize, index -> get(start + index));
    }

    /**
     * Returns a new Series containing elements from the beginning up to (but not including) the specified end index.
     *
     * @param exclusiveEnd The exclusive end index.
     * @return A new Series representing the head.
     */
    default @NotNull Series<T> head(int exclusiveEnd) {
        return slice(new IntRange(0, exclusiveEnd - 1)); // IntRange is inclusive, so -1
    }

    /**
     * Returns a new Series containing elements from the specified start index to the end.
     *
     * @param inclusiveStart The inclusive start index.
     * @return A new Series representing the tail.
     */
    default @NotNull Series<T> tail(int inclusiveStart) {
        return slice(new IntRange(inclusiveStart, size() - 1));
    }

    /**
     * Returns a new Series skipping the first 'n' elements.
     *
     * @param n The number of elements to skip.
     * @return A new Series with elements skipped.
     */
    default @NotNull Series<T> skip(int n) {
        return slice(new IntRange(n, size() - 1));
    }

    /**
     * Performs the given action for each element of the {@code Series}.
     *
     * @param action The action to be performed for each element.
     */
    default void each(@NotNull Consumer<T> action) {
        Objects.requireNonNull(action, "Action cannot be null.");
        for (int i = 0; i < size(); i++) {
            action.accept(get(i));
        }
    }

    /**
     * Converts this Series to an immutable List.
     * This operation materializes all elements.
     *
     * @return An immutable List containing all elements of this series.
     */
    default @NotNull List<T> toList() {
        return IntStream.range(0, size()).mapToObj(this::get).collect(Collectors.toUnmodifiableList());
    }

    /**
     * Returns the first element of the series.
     *
     * @return The first element.
     * @throws NoSuchElementException if the series is empty.
     */
    default T first() {
        if (size() == 0) throw new NoSuchElementException("Series is empty.");
        return get(0);
    }

    /**
     * Returns the last element of the series.
     *
     * @return The last element.
     * @throws NoSuchElementException if the series is empty.
     */
    default T last() {
        if (size() == 0) throw new NoSuchElementException("Series is empty.");
        return get(size() - 1);
    }

    /**
     * Checks if the series is empty.
     * @return true if the series contains no elements, false otherwise.
     */
    default boolean isEmpty() {
        return size() == 0;
    }

    @NotNull
    @Override
    default Iterator<T> iterator() {
        return new Iterator<>() {
            private int currentIndex = 0;

            @Override
            public boolean hasNext() {
                return currentIndex < size();
            }

            @Override
            public T next() {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }
                return get(currentIndex++);
            }
        };
    }

    /**
     * Immutable implementation of the Series interface.
     */
    final class ImmutableSeries<T> extends Join.ImmutableJoin<Integer, IntFunction<T>> implements Series<T> {
        private ImmutableSeries(int size, @NotNull IntFunction<T> provider) {
            super(size, provider);
        }

        @Override
        public int size() {
            return first();
        }

        @Override
        public T get(int index) {
            return second().apply(index);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            // For equality, we must compare elements, not just size and provider function reference.
            // This materializes the series for comparison, which can be expensive.
            Series<?> that = (Series<?>) o;
            if (this.size() != that.size()) return false;
            for (int i = 0; i < this.size(); i++) {
                if (!Objects.equals(this.get(i), that.get(i))) {
                    return false;
                }
            }
            return true;
        }

        @Override
        public int hashCode() {
            // Hash code based on elements, consistent with equals.
            // Materializes the series, can be expensive.
            return Objects.hash(toList());
        }

        @Override
        public String toString() {
            // For large series, this could be slow. Consider limiting elements.
            return "Series(" + size() + ")[" +
                   IntStream.range(0, Math.min(size(), 10)) // Limit for display
                           .mapToObj(this::get)
                           .map(Objects::toString)
                           .collect(Collectors.joining(", ")) +
                   (size() > 10 ? ", ..." : "") + "]";
        }
    }

    /**
     * Represents an inclusive integer range [from, to].
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
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            IntRange intRange = (IntRange) o;
            return from == intRange.from && to == intRange.to;
        }

        @Override
        public int hashCode() {
            return Objects.hash(from, to);
        }

        @Override
        public String toString() {
            return "[" + from + ", " + to + "]";
        }
    }
}
