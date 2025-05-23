package com.moneyfan.dsel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class ListBackedCursor<F, S> implements DSEL_Cursor<F, S> { // Added <F,S> for clarity
    // Underlying data store; effectively final after construction or private modification
    private List<Join<F, S>> data;
    private int cursorPosition; // For emulating head/tail without list modification for performance

    public ListBackedCursor(List<Join<F, S>> data) {
        this.data = Objects.requireNonNull(data, "Data list cannot be null.");
        this.cursorPosition = 0;
    }

    private ListBackedCursor(List<Join<F, S>> data, int cursorPosition) {
        this.data = data;
        this.cursorPosition = cursorPosition;
        if (this.data == null) {
            throw new IllegalArgumentException("Data list cannot be null.");
        }
    }

    public static <F, S> ListBackedCursor<F, S> of(List<Join<F, S>> list) {
        return new ListBackedCursor<>(list);
    }

    @SafeVarargs
    public static <F, S> ListBackedCursor<F, S> of(Join<F, S>... joins) {
        return new ListBackedCursor<>(List.of(joins));
    }

    @Override
    public boolean isEmpty() {
        return cursorPosition >= data.size();
    }

    @Override
    public Join<F, S> head() {
        if (isEmpty()) {
            throw new java.util.NoSuchElementException("Cursor is empty or past the end.");
        }
        return data.get(cursorPosition);
    }

    @Override
    public DSEL_Cursor<F, S> tail() {
        if (isEmpty()) {
            throw new java.util.NoSuchElementException("Cursor is empty.");
        }
        return new ListBackedCursor<>(new ArrayList<>(this.data), this.cursorPosition + 1);
    }

    @Override
    public Stream<Join<F, S>> stream() {
        return data.subList(cursorPosition, data.size()).stream();
    }

    @Override
    public List<Join<F, S>> collect() {
        return stream().collect(Collectors.toUnmodifiableList());
    }

    @Override
    public <R> ListBackedCursor<R, S> mapFirst(Function<? super F, ? extends R> fn) {
        List<Join<R, S>> mappedData = stream()
                .map(join -> JoinOps.j(fn.apply(join.first()), join.second()))
                .collect(Collectors.toList());
        return new ListBackedCursor<>(mappedData);
    }

    @Override
    public <R> ListBackedCursor<F, R> mapSecond(Function<? super S, ? extends R> fn) {
        List<Join<F, R>> mappedData = stream()
                .map(join -> JoinOps.j(join.first(), fn.apply(join.second())))
                .collect(Collectors.toList());
        return new ListBackedCursor<>(mappedData);
    }

    @Override
    public <R1, R2> ListBackedCursor<R1, R2> mapBoth(Function<? super F, ? extends R1> fn1, Function<? super S, ? extends R2> fn2) {
        List<Join<R1, R2>> mappedData = stream()
                .map(join -> JoinOps.j(fn1.apply(join.first()), fn2.apply(join.second())))
                .collect(Collectors.toList());
        return new ListBackedCursor<>(mappedData);
    }

    @Override
    public ListBackedCursor<S, F> swap() {
        List<Join<S, F>> swappedData = stream()
                .map(join -> JoinOps.j(join.second(), join.first()))
                .collect(Collectors.toList());
        return new ListBackedCursor<>(swappedData);
    }

    @Override
    public ListBackedCursor<F, S> filter(Predicate<? super Join<F, S>> predicate) {
        List<Join<F, S>> filteredData = stream()
                .filter(predicate)
                .collect(Collectors.toList());
        return new ListBackedCursor<>(filteredData);
    }

    @Override
    public Iterator<Join<F, S>> iterator() {
        return stream().iterator();
    }

    // Enum for shorthand operations / factory methods as requested
    public enum JoinOps {
        ; // No instances for this utility enum

        /**
         * Shorthand for creating a new Join. Glyph: j
         */
        public static <F, S> Join<F, S> j(F first, S second) {
            return new Join<>(first, second);
        }
    }
}
