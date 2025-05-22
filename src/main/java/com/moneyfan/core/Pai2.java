package com.moneyfan.core;

/**
 * An immutable pair (tuple of two elements).
 * @param <F> type of the first element
 * @param <S> type of the second element
 */
public record Pai2<F, S>(F first, S second) {

    /**
     * Static factory method for creating a new {@link Pai2}.
     */
    public static <F, S> Pai2<F, S> of(F first, S second) {
        return new Pai2<>(first, second);
    }
}