package com.moneyfan.dsel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;

class ListBackedCursor<F, S> implements DSEL_Cursor<F, S> {
    private final List<Join<F, S>> data;

    public ListBackedCursor(List<Join<F, S>> data) {
        // Ensure internal list is unmodifiable and a copy is made if mutable list is passed.
        this.data = Collections.unmodifiableList(new ArrayList<>(data));
    }
@Override
public <R> DSEL_Cursor<R, S> mapFirst(Function<? super F, ? extends R> mapper) {
    List<Join<R, S>> newData = data.stream()
        .map(join -> new Join<>(mapper.apply(join.first()), join.second()))
        .toList();
    return new ListBackedCursor<>(newData);
}

@Override
public <R> DSEL_Cursor<F, R> mapSecond(Function<? super S, ? extends R> mapper) {
    List<Join<F, R>> newData = data.stream()
        .map(join -> new Join<>(join.first(), mapper.apply(join.second())))
        .toList();
    return new ListBackedCursor<>(newData);
}
    @Override
    public <R1, R2> DSEL_Cursor<R1, R2> mapBoth(
        Function<? super F, ? extends R1> firstMapper,
        Function<? super S, ? extends R2> secondMapper) {
        List<Join<R1, R2>> newData = data.stream()
            .map(join -> new Join<>(firstMapper.apply(join.first()), secondMapper.apply(join.second())))
            .collect(Collectors.toList());
        return new ListBackedCursor<>(newData);
    }

    @Override
    public DSEL_Cursor<S, F> swap() {
        List<Join<S, F>> newData = data.stream()
            .map(Join::swap)
            .collect(Collectors.toList());
        return new ListBackedCursor<>(newData);
    }

    @Override
    public DSEL_Cursor<F, S> filter(Predicate<Join<F, S>> predicate) {
        List<Join<F, S>> newData = data.stream()
            .filter(predicate)
            .collect(Collectors.toList());
        return new ListBackedCursor<>(newData);
    }

    @Override
    public DSEL_Cursor<F, S> filterFirst(Predicate<? super F> predicate) {
        return filter(join -> predicate.test(join.getFirst()));
    }

    @Override
    public DSEL_Cursor<F, S> filterSecond(Predicate<? super S> predicate) {
        return filter(join -> predicate.test(join.getSecond()));
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
            throw new java.util.NoSuchElementException("Cursor is empty");
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
    public java.util.Iterator<Join<F, S>> iterator() {
        return data.iterator(); // The list itself is unmodifiable, so its iterator is safe.
    }
}
