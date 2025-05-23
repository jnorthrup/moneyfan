package com.moneyfan.dsel;

import java.util.function.Function;

/**
 * Core immutable 2-ary tuple (record).
 * Provides basic compositional operations.
 * F = First, S = Second
 */
public record Join<F, S>(F first, S second) {

    // Shorthand for mapFirst: mf
    public <R> Join<R, S> mf(Function<? super F, ? extends R> mapper) {
        return new Join<>(mapper.apply(first), second);
    }

    // Shorthand for mapSecond: ms
    public <R> Join<F, R> ms(Function<? super S, ? extends R> mapper) {
        return new Join<>(first, mapper.apply(second));
    }

    // Shorthand for swap: sw
    public Join<S, F> sw() { return new Join<>(second, first); }
}
