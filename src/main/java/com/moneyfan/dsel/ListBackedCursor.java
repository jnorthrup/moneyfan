package com.moneyfan.dsel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class ListBackedCursor<F, S> implements DSEL_Cursor<F, S> {

    private final List<Join<F, S>> data;

    public ListBackedCursor(List<Join<F, S>> data) {
        this.data = Objects.requireNonNull(data, "Data list cannot be null.");
    }

    @Override
    @SuppressWarnings("unchecked")
    public <R> DSEL_Cursor<R, S> mf(Function<? super F, ? extends R> mapper) {
        List<Join<R, S>> result = (List<Join<R, S>>) (List<?>) data.stream()
                .map(join -> join.mf(mapper))
                .collect(Collectors.toList());
        return new ListBackedCursor<>(result);
    }
    
    @Override
    @SuppressWarnings("unchecked")
    public <R> DSEL_Cursor<F, R> ms(Function<? super S, ? extends R> mapper) {
        List<Join<F, R>> result = (List<Join<F, R>>) (List<?>) data.stream()
                .map(join -> join.ms(mapper))
                .collect(Collectors.toList());
        return new ListBackedCursor<>(result);
    }

    @Override
    public <R1, R2> DSEL_Cursor<R1, R2> mb(BiFunction<? super F, ? super S, Join<R1, R2>> biMapper) {
        List<Join<R1, R2>> result = data.stream()
                .map(join -> biMapper.apply(join.first(), join.second()))
                .collect(Collectors.toList());
        return new ListBackedCursor<>(result);
    }

    @Override
    public DSEL_Cursor<F, S> fl(Predicate<Join<F, S>> predicate) {
        List<Join<F, S>> result = data.stream()
                .filter(predicate)
                .collect(Collectors.toList());
        return new ListBackedCursor<>(result);
    }

    @Override
    public DSEL_Cursor<F, S> head(int count) {
        int end = Math.min(count, data.size());
        return new ListBackedCursor<>(data.subList(0, end));
    }

    @Override
    public DSEL_Cursor<F, S> tail(int count) {
        int start = Math.min(count, data.size());
        return new ListBackedCursor<>(data.subList(start, data.size()));
    }

    @Override
    public DSEL_Cursor<F, S> skip(int count) {
        int start = Math.min(count, data.size());
        return new ListBackedCursor<>(data.subList(start, data.size()));
    }

    @Override
    public List<Join<F, S>> collect() {
        return new ArrayList<>(data);
    }

    @Override
    public Stream<Join<F, S>> stream() {
        return data.stream();
    }

    @Override
    public DSEL_Cursor<F, S> print(int count) {
        data.stream().limit(Math.max(0, count)).forEach(System.out::println);
        return this;
    }

    @Override
    public long count() {
        return data.size();
    }

    @Override
    public Iterator<Join<F, S>> iterator() {
        return data.iterator();
    }

    @Override
    public boolean isEmp() {
        return data.isEmpty();
    }

    @Override
    public java.util.Optional<Join<F, S>> fstJ() {
        return data.isEmpty() ? java.util.Optional.empty() : java.util.Optional.of(data.get(0));
    }
}
