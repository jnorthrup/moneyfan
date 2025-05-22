package com.moneyfan.core;

/**
 * Immutable pair record similar to Kotlin's Pair.
 * @param <F> first element type
 * @param <S> second element type
 */
public record Pai2<F, S>(F first, S second) {

    /**
     * Static factory method with type inference.
     */
    public static <F, S> Pai2<F, S> of(F first, S second) {
        return new Pai2<>(first, second);
    }
}