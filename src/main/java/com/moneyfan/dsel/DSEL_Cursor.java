package com.moneyfan.dsel;

import java.util.List;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * Abstract interface for a cursor over a sequence of {@link Join} tuples.
 * This forms the core of the DSEL, providing operations inspired by Pandas Series/DataFrames
 * but strictly adhering to the 2-ary tuple composition.
 *
 * All operations are designed to be unary from the perspective of the cursor
 * (i.e., they take one cursor and produce one cursor or a terminal value),
 * promoting compositional purity.
 */
public interface DSEL_Cursor<F, S> extends Iterable<Join<F, S>> {

    // --- Core Transformation Operations ---

    /** Maps the first element of each Join tuple. Glyph: mfst */
    <R> DSEL_Cursor<R, S> mapFirst(Function<? super F, ? extends R> mapper);
    default <R> DSEL_Cursor<R, S> mfst(Function<? super F, ? extends R> mapper) { return mapFirst(mapper); }

    /** Maps the second element of each Join tuple. Glyph: msnd */
    <R> DSEL_Cursor<F, R> mapSecond(Function<? super S, ? extends R> mapper);
    default <R> DSEL_Cursor<F, R> msnd(Function<? super S, ? extends R> mapper) { return mapSecond(mapper); }

    /** Maps both elements of each Join tuple. Glyph: mbth */
    <R1, R2> DSEL_Cursor<R1, R2> mapBoth(
        Function<? super F, ? extends R1> firstMapper,
        Function<? super S, ? extends R2> secondMapper
    );
    default <R1, R2> DSEL_Cursor<R1, R2> mbth(
        Function<? super F, ? extends R1> firstMapper,
        Function<? super S, ? extends R2> secondMapper
    ) { return mapBoth(firstMapper, secondMapper); }

    /** Swaps the elements of each Join tuple. Glyph: swp */
    DSEL_Cursor<S, F> swap();
    default DSEL_Cursor<S, F> swp() { return swap(); }

    // --- Filtering Operations ---

    /** Filters the cursor based on a predicate applied to the whole Join tuple. Glyph: flt */
    DSEL_Cursor<F, S> filter(Predicate<Join<F, S>> predicate);
    default DSEL_Cursor<F, S> flt(Predicate<Join<F, S>> predicate) { return filter(predicate); }

    /** Filters the cursor based on a predicate applied to the first element. Glyph: fltFst */
    DSEL_Cursor<F, S> filterFirst(Predicate<? super F> predicate);
    default DSEL_Cursor<F, S> fltFst(Predicate<? super F> predicate) { return filterFirst(predicate); }

    /** Filters the cursor based on a predicate applied to the second element. Glyph: fltSnd */
    DSEL_Cursor<F, S> filterSecond(Predicate<? super S> predicate);
    default DSEL_Cursor<F, S> fltSnd(Predicate<? super S> predicate) { return filterSecond(predicate); }

    // --- Terminal Operations ---

    /** Collects all Join tuples into a List. Glyph: col */
    List<Join<F, S>> collect();
    default List<Join<F, S>> col() { return collect(); }

    /** Counts the number of Join tuples in the cursor. Glyph: cnt */
    long count();
    default long cnt() { return count(); }

    /** Returns the first Join tuple, or throws if empty. Glyph: fstJ / headJ */
    Join<F,S> firstJoin();
    default Join<F,S> fstJ() { return firstJoin(); }

    /** Returns the first N Join tuples as a new cursor. Glyph: headC */
    DSEL_Cursor<F,S> head(int n);

    /** Returns the last N Join tuples as a new cursor. Glyph: tailC */
    DSEL_Cursor<F,S> tail(int n);

    // --- Utility ---

    /**
     * Prints a sample of the cursor to System.out for debugging.
     * Glyph: prn / peek
     */
    DSEL_Cursor<F, S> print(int N);
    default DSEL_Cursor<F, S> prn(int N) { return print(N); }
    default DSEL_Cursor<F, S> peek(int N) { return print(N); }

    /**
     * Checks if the cursor is empty. Glyph: isEmp
     */
    boolean isEmpty();
    default boolean isEmp() { return isEmpty(); }
}
