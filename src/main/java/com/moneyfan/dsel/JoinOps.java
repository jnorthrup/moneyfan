package com.moneyfan.dsel;

import java.util.function.Function;
import java.util.function.UnaryOperator;

/**
 * Enum acting as a bag of operations (unary operators or functions on Joins).
 * Houses logic, adhering to the compositional purity and simplicity mantra.
 * Uses short, unambiguous root names for potential IntelliJ completion.
 */
public enum JoinOps {
    // Singleton instance, can hold static methods or be used for type safety if needed.
    // For now, it's primarily a namespace for static utility methods.
    INSTANCE;

    // --- Shorthand Unary Operators ---

    // SWap Pair elements
    public static <F, S> UnaryOperator<Join<F, S>> swp() {
        return j -> j.sw(); // Delegates to Join's own swap method
    }

    // --- Shorthand Function-based Mappers ---

    // Map First element Of Join: mfoj
    public static <F, S, R> Function<Join<F, S>, Join<R, S>> mfoj(Function<? super F, ? extends R> fmap) {
        return j -> j.mf(fmap); // Delegates to Join's own mapFirst (mf)
    }

    // Map Second element Of Join: msoj
    public static <F, S, R> Function<Join<F, S>, Join<F, R>> msoj(Function<? super S, ? extends R> smap) {
        return j -> j.ms(smap); // Delegates to Join's own mapSecond (ms)
    }

    // Create Join: cj (utility for conciseness)
    public static <F, S> Join<F, S> cj(F first, S second) { return new Join<>(first, second); }
}
