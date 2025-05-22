package com.moneyfan.core;

/**
 * Immutable pair record.
 */
public record Pai2<F, S>(F first, S second) {
    
    public static <F, S> Pai2<F, S> of(F first, S second) {
        return new Pai2<>(first, second);
    }
}