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

public class ListBackedCursor<F, S> implements DSEL_Cursor<F, S> {
    private List<Join<F, S>> data;
    private int cursorPosition;

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
    public Join<F, S> head(int index) {
        int targetPosition = cursorPosition + index;
        if (targetPosition < cursorPosition || targetPosition >= data.size()) {
            throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for cursor at position " + cursorPosition + " with size " + (data.size() - cursorPosition));
        }
        return data.get(targetPosition);
    }

    @Override
    public DSEL_Cursor<F, S> tail(int count) {
        if (count < 0) {
            throw new IllegalArgumentException("Count for tail cannot be negative.");
        }
        int newPosition = cursorPosition + count + 1;  // +1 because tail is "after"
        if (newPosition > data.size()) {
            newPosition = data.size();  // Effectively an empty cursor
        }
        return new ListBackedCursor<>(this.data, newPosition);
    }

    @Override
    public void print(int i) {
        System.out.println("Printing next " + i + " elements from cursor:");
        List<Join<F, S>> toPrint = stream().limit(i).collect(Collectors.toList());
        if (toPrint.isEmpty()) {
            System.out.println("(No elements to print or cursor is empty)");
        } else {
            toPrint.forEach(join -> System.out.println(JoinOps.str(join)));
        }
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

    public Stream<Join<F, S>> stream() {
        if (isEmpty()) {
            return Stream.empty();
        }
        return data.subList(cursorPosition, data.size()).stream();
    }

    public List<Join<F, S>> toList() {
        return stream().collect(Collectors.toUnmodifiableList());
    }

    public enum JoinOps {
        ;

        public static <F, S> Join<F, S> j(F first, S second) {
            return new Join<>(first, second);
        }

        public static <F, S> String str(Join<F, S> join) {
            return (join == null) ? "null" : join.first() + " :: " + join.second();
        }
    }
}
