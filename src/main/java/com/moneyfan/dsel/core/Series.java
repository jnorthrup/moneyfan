package com.moneyfan.dsel.core;

import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Represents a Series of elements, defined as a Join of an Integer count and a Function
 * that provides elements by their Integer index. This structure enables lazy access
 * and aligns with cursor-based data retrieval (e.g., for ISAM/mmap data).
 *
 * @param <T> The type of elements in the Series.
 */
public record Series<T>(Integer size, Function<Integer, T> accessor) implements Join<Integer, Function<Integer, T>> {

    // Aliases for clarity when accessing Series components
    public Integer size() { return size; }
    public Function<Integer, T> accessor() { return accessor; }

    /**
     * Factory method for creating a Series.
     *
     * @param size The total number of elements in the series.
     * @param accessor A function to retrieve an element by its 0-based index.
     * @param <T> The type of the elements.
     * @return A new Series instance.
     */
    public static <T> Series<T> ser(int size, Function<Integer, T> accessor) {
        return new Series<>(size, accessor);
    }

    /**
     * Creates a Series backed by a List. This is a convenience for initial data loading
     * or for scenarios where data is already materialized.
     *
     * @param list The list to back the series.
     * @param <T> The type of the elements in the list.
     * @return A new Series instance.
     */
    public static <T> Series<T> ser(List<T> list) {
        Objects.requireNonNull(list, "List for Series cannot be null.");
        return new Series<>(list.size(), list::get);
    }

    // --- Core Series operations, operating on the 'T' values within the series ---

    /**
     * Applies a function to each value in the Series, creating a new Series with the transformed values.
     * This operation is lazy; values are transformed only when accessed via the new Series' accessor.
     *
     * @param mapper The function to apply to each element.
     * @param <R> The type of the new elements.
     * @return A new Series with transformed elements.
     */
    public <R> Series<R> mapVal(Function<? super T, ? extends R> mapper) {
        return new Series<>(size(), i -> mapper.apply(accessor().apply(i)));
    }

    /**
     * Filters the Series based on a predicate applied to each value.
     * This operation will materialize a new list of *indices* that satisfy the predicate.
     * The resulting Series' accessor will then map these new indices to the original
     * accessor using the stored filtered indices. This provides a balance between
     * laziness and maintaining contiguous indexing for the filtered view.
     *
     * @param predicate The predicate to test each element.
     * @return A new Series containing only the elements that satisfy the predicate.
     */
    public Series<T> filterVal(Predicate<? super T> predicate) {
        List<Integer> filteredIndices = IntStream.range(0, size())
                                                .filter(i -> predicate.test(accessor().apply(i)))
                                                .boxed()
                                                .toList();
        return new Series<>(filteredIndices.size(), i -> accessor().apply(filteredIndices.get(i)));
    }

    /**
     * Returns a new Series containing the first 'n' elements.
     *
     * @param n The number of elements to take from the head.
     * @return A new Series representing the head.
     */
    public Series<T> head(int n) {
        int newSize = Math.min(n, size());
        return new Series<>(newSize, accessor()); // Accessor is already relative to the original Series' start
    }

    /**
     * Returns a new Series containing the last 'n' elements.
     *
     * @param n The number of elements to take from the tail.
     * @return A new Series representing the tail.
     */
    public Series<T> tail(int n) {
        int actualN = Math.min(n, size());
        int startIndex = size() - actualN;
        return new Series<>(actualN, i -> accessor().apply(startIndex + i));
    }

    /**
     * Skips the first 'n' elements and returns a new Series with the remaining elements.
     *
     * @param n The number of elements to skip.
     * @return A new Series representing the remaining elements.
     */
    public Series<T> skip(int n) {
        int startIndex = Math.min(n, size());
        int newSize = size() - startIndex;
        return new Series<>(newSize, i -> accessor().apply(startIndex + i));
    }

    /**
     * Retrieves the element at the specified 0-based index.
     *
     * @param index The index of the element to retrieve.
     * @return The element at the given index.
     * @throws IndexOutOfBoundsException if the index is out of bounds.
     */
    public T at(int index) {
        if (index < 0 || index >= size()) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for Series of size " + size());
        }
        return accessor().apply(index);
    }

    /**
     * Checks if the Series is empty.
     * @return true if the Series contains no elements, false otherwise.
     */
    public boolean isEmp() {
        return size() == 0;
    }

    /**
     * Returns an Optional containing the first element of the Series, or empty if the Series is empty.
     * @return An Optional containing the first element.
     */
    public java.util.Optional<T> fstV() {
        return isEmp() ? java.util.Optional.empty() : java.util.Optional.of(accessor().apply(0));
    }

    /**
     * Prints the first 'n' elements of the Series to standard output.
     * @param n The number of elements to print.
     * @return The Series itself for chaining.
     */
    public Series<T> print(int n) {
        IntStream.range(0, Math.min(n, size()))
                .mapToObj(i -> accessor().apply(i))
                .forEach(System.out::println);
        return this;
    }

    /**
     * Returns a Stream over the elements of the Series.
     * @return A Stream of elements.
     */
    public Stream<T> streamVal() {
        return IntStream.range(0, size()).mapToObj(i -> accessor().apply(i));
    }

    /**
     * Collects all elements into a List. This materializes the entire Series.
     * Use with caution for very large Series.
     * @return A List of all elements.
     */
    public List<T> collectVal() {
        return streamVal().collect(Collectors.toList());
    }

    /**
     * Provides an Iterable over the elements of the Series for transitioning out of the Series context.
     * @return An Iterable of elements.
     */
    public Iterable<T> iteratable() {
        return new Iterable<>() {
            @Override
            public Iterator<T> iterator() {
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
                        return accessor().apply(currentIndex++);
                    }
                };
            }
        };
    }

    @Override
    public Integer first() {
        return size;
    }

    @Override
    public Function<Integer, T> second() {
        return accessor;
    }

    @Override
    public <R> Join<R, Function<Integer, T>> mapFirst(Function<? super Integer, ? extends R> func) {
        return new JoinImpl<>(func.apply(size), accessor);
    }

    @Override
    public <R> Join<Integer, R> mapSecond(Function<? super Function<Integer, T>, ? extends R> func) {
        return new JoinImpl<>(size, func.apply(accessor));
    }

    @Override
    public Join<Function<Integer, T>, Integer> swap() {
        return new JoinImpl<>(accessor, size);
    }

    @Override
    public <R1, R2> Join<R1, R2> bimap(Function<? super Integer, ? extends R1> fMap, Function<? super Function<Integer, T>, ? extends R2> sMap) {
        return new JoinImpl<>(fMap.apply(size), sMap.apply(accessor));
    }
}