package com.moneyfan.dsel.core;

import java.util.function.Function;

/**
 * An immutable 2-ary tuple, serving as the core data structure for the DSEL.
 *
 * @param <F> The type of the first element.
 * @param <S> The type of the second element.
 */
public record Join<F, S>(F fst, S snd) {

    /**
     * Static factory method for creating Join instances.
     * Provides a concise way to create Joins, e.g., via static import.
     */
    public static <F, S> Join<F, S> jn(F first, S second) {
        return new Join<>(first, second);
    }

    /**
     * Applies a function to the first element of this Join.
     *
     * @param func The function to apply.
     * @param <R>  The type of the result of the function.
     * @return A new Join with the transformed first element and the original second element.
     */
    public <R> Join<R, S> mapFst(Function<? super F, ? extends R> func) {
        return new Join<>(func.apply(fst), snd);
    }

    /**
     * Applies a function to the second element of this Join.
     *
     * @param func The function to apply.
     * @param <R>  The type of the result of the function.
     * @return A new Join with the original first element and the transformed second element.
     */
    public <R> Join<F, R> mapSnd(Function<? super S, ? extends R> func) {
        return new Join<>(fst, func.apply(snd));
    }

    /**
     * Swaps the elements of this Join.
     *
     * @return A new Join with the elements swapped.
     */
    public Join<S, F> swap() {
        return new Join<>(snd, fst);
    }

    /**
     * "Curries" this Join, applying a function that takes the first element
     * and returns a function that takes the second element.
     * This is a conceptual representation; true currying is function-level.
     * Here, it can be used to transform the Join based on both elements.
     */
    public <R> R curry(Function<? super F, Function<? super S, ? extends R>> func) {
        return func.apply(fst).apply(snd);
    }
}
