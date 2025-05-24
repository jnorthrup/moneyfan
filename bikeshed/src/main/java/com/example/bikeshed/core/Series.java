package com.example.bikeshed.core;

import com.example.bikeshed.dsel.D;
import com.example.bikeshed.dsel.Join;

import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.function.Predicate;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

/**
 * Represents a cursor-based collection, conceptually a `Join<Integer, Function<Integer, T>>`.
 * This structure is optimized for ISAM data access patterns (e.g., get(index), slice(range)),
 * supporting memory-mapped files (mmap) for efficient time-series data access.
 * It emphasizes *phased adoption* where `Series` replaces `List`-based workflows.
 *
 * This class serves as the core, immutable Series implementation.
 * The DSEL layer (`com.example.bikeshed.dsel.Series`) will extend or wrap this.
 *
 * @param <T> The type of elements in the series.
 */
public class Series<T> extends Join<Integer, IntFunction<T>> implements Iterable<T> {

    protected Series(Integer size, IntFunction<T> provider) {
        super(size, provider);
    }

    /**
     * Factory method for creating a Series instance.
     * This method is typically accessed via the DSEL utility enum (e.g., D.sr(size, provider)).
     *
     * @param size The number of elements in the series.
     * @param provider A function that provides an element given its index.
     * @param <T> The type of elements.
     * @return A new immutable Series instance.
     */
    public static <T> Series<T> of(int size, IntFunction<T> provider) {
        return new Series<>(size, provider);
    }

    /**
     * Factory method to create an empty Series.
     * @param <T> The type of elements.
     * @return An empty Series.
     */
    public static <T> Series<T> empty() {
        return new Series<>(0, i -> { throw new NoSuchElementException("Accessed empty Series"); });
    }

    /**
     * Gets an element by its index.
     * Provides "operator overloading via convention" for `get(index)`.
     *
     * @param index The index of the element.
     * @return The element at the specified index.
     * @throws IndexOutOfBoundsException if the index is out of bounds.
     */
    public T get(int index) {
        if (index < 0 || index >= size()) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for Series of size " + size());
        }
        return b().apply(index);
    }

    /**
     * Returns the size of the series.
     *
     * @return The number of elements.
     */
    public int size() {
        return a();
    }

    public boolean isEmpty() {
        return size() == 0;
    }

    /**
     * Slices the series to a new sub-series.
     * Compositional: returns a new Series instance.
     *
     * @param startIndex Inclusive start index.
     * @param endIndex Exclusive end index.
     * @return A new Series representing the slice.
     */
    public Series<T> slice(int startIndex, int endIndex) {
        if (startIndex < 0 || startIndex >= size() || endIndex > size() || startIndex > endIndex) {
            throw new IndexOutOfBoundsException("Invalid slice range: [" + startIndex + ", " + endIndex + ") for size " + size());
        }
        int newSize = endIndex - startIndex;
        return D.sr(newSize, i -> get(startIndex + i));
    }

    /**
     * Slices the series using an IntRange.
     * Provides "operator overloading via convention" for `slice(range)`.
     *
     * @param range The range of indices.
     * @return A new Series representing the slice.
     */
    public Series<T> get(java.util.function.IntPredicate range) { // This is a placeholder for `get(IntRange)`
        // In Java, direct operator overloading for `IntRange` isn't available.
        // We can either provide a concrete `IntRange` equivalent or expose a method.
        // For now, let's assume `get(int startIndex, int endIndex)` as the primary slicing.
        // A more advanced DSEL could parse a String "0..10" for ranges.
        throw new UnsupportedOperationException("Predicate-based range slicing not directly supported for performance. Use explicit start/end indices.");
    }

    /**
     * Maps each element of the Series to a new type.
     * Compositional: returns a new Series instance.
     *
     * @param mapper Function to apply to each element.
     * @param <R>    New type of elements.
     * @return A new Series with transformed elements.
     */
    public <R> Series<R> map(Function<? super T, ? extends R> mapper) {
        return D.sr(size(), i -> mapper.apply(get(i)));
    }

    /**
     * Filters elements of the Series based on a predicate.
     * Compositional: returns a new Series instance (may be smaller).
     * Note: For lazy evaluation, this might build an intermediate list of indices or a lazy stream.
     * For high-performance, an actual implementation might return a Series that filters on-access.
     * For now, a simpler list-based approach is used for illustration.
     *
     * @param predicate Predicate to test each element.
     * @return A new Series containing only elements that satisfy the predicate.
     */
    public Series<T> filter(Predicate<? super T> predicate) {
        // This is a simple implementation that materializes filtered elements.
        // For large datasets, a true lazy filter (like a filtering iterator or a Series
        // that wraps the original and applies the predicate on get) would be more efficient.
        List<T> filteredElements = IntStream.range(0, size())
                .mapToObj(this::get)
                .filter(predicate)
                .collect(java.util.ArrayList::new, java.util.ArrayList::add, java.util.ArrayList::addAll);

        return D.sr(filteredElements.size(), filteredElements::get);
    }

    /**
     * Reduces the Series to a single value using a combining function.
     *
     * @param identity The initial value of the accumulation.
     * @param accumulator A function that combines an accumulated result and an element.
     * @param <U> The type of the accumulated result.
     * @return The result of the reduction.
     */
    public <U> U reduce(U identity, java.util.function.BiFunction<U, ? super T, U> accumulator) {
        U result = identity;
        for (int i = 0; i < size(); i++) {
            result = accumulator.apply(result, get(i));
        }
        return result;
    }

    /**
     * Converts the Series to a standard Java List.
     * Note: This materializes the entire series into memory. Use with caution for large series.
     *
     * @return A List containing all elements of the Series.
     */
    public List<T> toList() {
        return IntStream.range(0, size())
                .mapToObj(this::get)
                .collect(java.util.ArrayList::new, java.util.ArrayList::add, java.util.ArrayList::addAll);
    }

    @Override
    public Iterator<T> iterator() {
        return new Iterator<T>() {
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
     * Provides a Stream API equivalent for Series.
     * Note: This method effectively materializes elements as they are streamed.
     *
     * @return A Stream of elements from this Series.
     */
    public Stream<T> stream() {
        return StreamSupport.stream(this.spliterator(), false);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Series<?> series = (Series<?>) o;
        if (size() != series.size()) return false;
        // Deep equality check for elements
        for (int i = 0; i < size(); i++) {
            if (!Objects.equals(this.get(i), series.get(i))) {
                return false;
            }
        }
        return true;
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(size());
        // Simple hash code. For large series, consider a sampling or a more efficient approach.
        // For DSEL, immutability helps with memoization of hash codes.
        for (int i = 0; i < size(); i++) {
            result = 31 * result + Objects.hashCode(get(i));
        }
        return result;
    }

    @Override
    public String toString() {
        return "Series(size=" + size() + ")";
    }
}
