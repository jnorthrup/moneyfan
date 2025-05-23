package com.moneyfan.dsel;

import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Stream;

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

    // --- Core Operations ---

    // Map First element: mf
    <R> DSEL_Cursor<R, S> mf(Function<? super F, ? extends R> mapper);

    // Map Second element: ms
    <R> DSEL_Cursor<F, R> ms(Function<? super S, ? extends R> mapper);

    // Map Both elements (BiFunction): mb
    <R1, R2> DSEL_Cursor<R1, R2> mb(BiFunction<? super F, ? super S, Join<R1, R2>> biMapper);

    // Filter: fl
    DSEL_Cursor<F, S> fl(Predicate<Join<F, S>> predicate);

    // --- Positional Operations ---

    // Take N elements from the start: tk
    DSEL_Cursor<F, S> head(int count);

    // Take N elements from the end: tl
    DSEL_Cursor<F, S> tail(int count);

    // Skip N elements: sk
    DSEL_Cursor<F, S> skip(int count);

    // --- Terminal Operations ---

    // Collect to List: cl
    List<Join<F, S>> collect();

    // Collect to Stream: cs
    Stream<Join<F, S>> stream();

    // --- Utility/Debug Operations ---

    // Print N elements: pr
    DSEL_Cursor<F, S> print(int count);

    // Count elements: ct
    long count();

    // Swap elements in each Join: swp
    default DSEL_Cursor<S, F> swp() {
        return this.mb((f, s) -> JoinOps.cj(s, f));
    }

    // Check if cursor is empty: isEmp
    boolean isEmp();

    // Get first Join element: fstJ (returns Optional to handle empty cursors)
    java.util.Optional<Join<F, S>> fstJ();
}
