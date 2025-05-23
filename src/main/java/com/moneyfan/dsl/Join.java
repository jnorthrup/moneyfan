package com.moneyfan.dsl;

/**
 * The core 2-ary tuple. Immutable.
 * This is the fundamental building block for compositional data structures.
 *
 * @param <F> Type of the first element.
 * @param <S> Type of the second element.
 */
public record Join<F, S>(F first, S second) {
    /**
     * Glyph shorthand factory method for creating Join instances.
     * j(first, second)
     */
    public static <F, S> Join<F, S> j(F first, S second) {
        return new Join<>(first, second);
    }

    // Unary operators for projection
    public F fst() { return first; }
    public S snd() { return second; }
}
