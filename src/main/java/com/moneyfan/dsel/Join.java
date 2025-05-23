package com.moneyfan.dsel;

import java.util.function.Function;

/**
 * The core immutable 2-ary tuple (Pair) for the DSEL.
 * This record is the fundamental data structure. All operations in the DSEL
 * will revolve around consuming and producing Joins or Cursors of Joins.
 *
 * It serves as a "voluntary metaclass" in the sense that its instances are
 * the primary citizens of the DSEL, and the DSEL defines how they are manipulated.
 *
 * @param <F> Type of the first element.
 * @param <S> Type of the second element.
 */
public record Join<F, S>(F first, S second) {

    /** Static factory method for convenience. */
    public static <F_TYPE, S_TYPE> Join<F_TYPE, S_TYPE> of(F_TYPE first, S_TYPE second) {
        return new Join<>(first, second);
    }

    /** Swaps the elements of the tuple, returning a new Join. */
    public Join<S, F> swap() {
        return new Join<>(second, first);
    }

    /** Applies a function to the first element, returning a new Join. */
    public <R> Join<R, S> mapFirst(Function<? super F, ? extends R> mapper) {
        return new Join<>(mapper.apply(first), second);
    }

    /** Applies a function to the second element, returning a new Join. */
    public <R> Join<F, R> mapSecond(Function<? super S, ? extends R> mapper) {
        return new Join<>(first, mapper.apply(second));
    }

    /** Applies functions to both elements, returning a new Join. */
    public <R1, R2> Join<R1, R2> mapBoth(
        Function<? super F, ? extends R1> firstMapper,
        Function<? super S, ? extends R2> secondMapper) {
        return new Join<>(firstMapper.apply(first), secondMapper.apply(second));
    }
}
