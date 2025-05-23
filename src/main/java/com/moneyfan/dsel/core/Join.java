package com.moneyfan.dsel.core;

import java.util.Objects;

/**
 * An immutable 2-ary tuple (pair).
 * This is the foundational data structure for the DSEL.
 *
 * @param <F> the type of the first element
 * @param <S> the type of the second element
 */
public record Join<F, S>(F f, S s) {

    /**
     * Concise factory method for creating a Join.
     * Glyph shorthand: jn
     */
    public static <F, S> Join<F, S> jn(F first, S second) {
        return new Join<>(first, second);
    }
    // Custom equals, hashCode, and toString are implicitly provided by records.
}
