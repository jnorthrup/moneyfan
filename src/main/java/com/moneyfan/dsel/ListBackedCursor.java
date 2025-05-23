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
        List<Join<R, S>> newList = data.stream()
            .map(join -> new Join<>(mapper.apply(join.first()), join.second()))
            .collect(Collectors.toList());
        return new ListBackedCursor<>(newList);
    }

    @Override
    public <R> DSEL_Cursor<F, R> mapSecond(Function<? super S, ? extends R> mapper) {
        List<Join<F, R>> newList = data.stream()
            .map(join -> new Join<>(join.first(), mapper.apply(join.second())))
            .collect(Collectors.toList());
        return new ListBackedCursor<>(newList);
    }

    @Override
    public <R1, R2> DSEL_Cursor<R1, R2> mapBoth(
        Function<? super F, ? extends R1> firstMapper,
        Function<? super S, ? extends R2> secondMapper) {
        List<Join<R1, R2>> newList = data.stream()
            .map(join -> new Join<>(firstMapper.apply(join.first()), secondMapper.apply(join.second())))
            .collect(Collectors.toList());
        return new ListBackedCursor<>(newList);
    }

    @Override
    public DSEL_Cursor<S, F> swap() {
        List<Join<S, F>> newList = data.stream()
            .map(Join::swap)
            .collect(Collectors.toList());
        return new ListBackedCursor<>(newList);
    }

    @Override
    public DSEL_Cursor<F, S> filter(Predicate<Join<F, S>> predicate) {
        List<Join<F, S>> newList = data.stream()
            .filter(predicate)
            .collect(Collectors.toList());
        return new ListBackedCursor<>(newList);
    }

    @Override
    public DSEL_Cursor<F, S> filterFirst(Predicate<? super F> predicate) {
        return filter(join -> predicate.test(join.first()));
    }

    @Override
    public DSEL_Cursor<F, S> filterSecond(Predicate<? super S> predicate) {
        return filter(join -> predicate.test(join.second()));
    }

    public List<Join<F, S>> toList() {
        return data; // Already an immutable copy from constructor
    }

    /** Gets a list of all first elements. Addresses: {@code [60,50] cannot find symbol getFirst()}. */
    public List<F> getFirsts() { // Renamed from getFirstColumn for clarity
        return data.stream()
                   .map(Join::first) // Line 60: Uses record accessor `first()`.
                    .collect(Collectors.toList());
    }

    /** Gets a list of all second elements. Addresses: {@code [65,50] cannot find symbol getSecond()}. */
    public List<S> getSeconds() { // Renamed from getSecondColumn for clarity
        return data.stream()
                   .map(Join::second) // Line 65: Uses record accessor `second()`.
                    .collect(Collectors.toList());
    }

    // Glyphs (shorthands) as per request - examples
    /** Glyph for {@link #mapFirst(Function)}. */
    public <R> ListBackedCursor<R, S> mF(Function<? super F, ? extends R> fn) { return mapFirst(fn); }
    /** Glyph for {@link #mapSecond(Function)}. */
    public <R> ListBackedCursor<F, R> mS(Function<? super S, ? extends R> fn) { return mapSecond(fn); }
    /** Glyph for {@link #swap()}. */
    public ListBackedCursor<S, F> swA() { return swap(); }
    /** Glyph for {@link #mapBoth(Function, Function)}. */
    public <R1, R2> ListBackedCursor<R1, R2> mJ(Function<? super Join<F, S>, ? extends Join<R1, R2>> fn) { return mapBoth(fn); }  // Adjusted based on context
    /** Glyph for {@link #getFirsts()}. */
    public List<F> fL() { return getFirsts(); } // Firsts List
    /** Glyph for {@link #getSeconds()}. */
    public List<S> sL() { return getSeconds(); } // Seconds List
}
