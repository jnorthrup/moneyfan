package com.moneyfan.dsel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;

/**
 * A basic, immutable, list-backed implementation of {@link DSEL_Cursor}.
 * Operations create new instances, preserving immutability and enabling composition.
 */
class ListBackedCursor<F, S> implements DSEL_Cursor<F, S> {
    private final List<Join<F, S>> data;

    public ListBackedCursor(List<Join<F, S>> data) {
        // Ensure internal list is unmodifiable and a copy is made if mutable list is passed.
        this.data = Collections.unmodifiableList(new ArrayList<>(data));
    }

    @Override
    public <R> DSEL_Cursor<R, S> mapFirst(Function<? super F, ? extends R> mapper) {
        return new ListBackedCursor<R, S>(data.stream()
            .map(join -> join.mapFirst(mapper))
            .collect(Collectors.toList()));
    }

    @Override
    public <R> DSEL_Cursor<F, R> mapSecond(Function<? super S, ? extends R> mapper) {
        return new ListBackedCursor<F, R>(data.stream()
            .map(join -> join.mapSecond(mapper))
            .collect(Collectors.toList()));
    }

    @Override
    public <R1, R2> DSEL_Cursor<R1, R2> mapBoth(
        Function<? super F, ? extends R1> firstMapper,
        Function<? super S, ? extends R2> secondMapper) {
        return new ListBackedCursor<R1, R2>(data.stream()
            .map(join -> join.mapBoth(firstMapper, secondMapper))
            .collect(Collectors.toList()));
    }

    @Override
    public DSEL_Cursor<S, F> swap() {
        return new ListBackedCursor<S, F>(data.stream()
            .map(Join::swap)
            .collect(Collectors.toList()));
    }

    @Override
    public DSEL_Cursor<F, S> filter(Predicate<Join<F, S>> predicate) {
        return new ListBackedCursor<>(data.stream()
            .filter(predicate)
            .collect(Collectors.toList()));
    }

    @Override
    public DSEL_Cursor<F, S> filterFirst(Predicate<? super F> predicate) {
        return filter(join -> predicate.test(join.first()));
    }

    @Override
    public DSEL_Cursor<F, S> filterSecond(Predicate<? super S> predicate) {
        return filter(join -> predicate.test(join.second()));
    }

    @Override
    public List<Join<F, S>> collect() {
        return new ArrayList<>(data); // Return a mutable copy as per typical collect behavior
    }

    @Override
    public long count() {
        return data.size();
    }

    @Override
    public Join<F, S> firstJoin() {
        if (data.isEmpty()) {
            throw new NoSuchElementException("Cursor is empty");
        }
        return data.get(0);
    }

    @Override
    public DSEL_Cursor<F, S> head(int n) {
        if (n < 0) throw new IllegalArgumentException("N must be non-negative");
        return new ListBackedCursor<>(data.subList(0, Math.min(n, data.size())));
    }

    @Override
    public DSEL_Cursor<F, S> tail(int n) {
        if (n < 0) throw new IllegalArgumentException("N must be non-negative");
        return new ListBackedCursor<>(data.subList(Math.max(0, data.size() - n), data.size()));
    }

    @Override
    public Iterator<Join<F, S>> iterator() {
        return data.iterator(); // The list itself is unmodifiable, so its iterator is safe.
    }

    @Override
    public DSEL_Cursor<F, S> print(int N) {
        System.out.println("Cursor (showing up to " + N + " elements):");
        if (data.isEmpty()) {
            System.out.println("  <empty>");
        } else {
            data.stream().limit(N).forEach(join -> System.out.println("  " + join));
            if (data.size() > N) {
                System.out.println("  ... and " + (data.size() - N) + " more");
            }
        }
        return this; // Return self to allow chaining
    }

    @Override
    public boolean isEmpty() {
        return data.isEmpty();
    }

    @Override
    public String toString() {
        if (data.isEmpty()) {
            return "ListBackedCursor[size=0]";
        } else if (data.size() == 1) {
            return "ListBackedCursor[size=1, head=" + data.get(0) + "]";
        } else {
            return "ListBackedCursor[size=" + data.size() + ", head=" + data.get(0) + ", ...]";
        }
    }
}
