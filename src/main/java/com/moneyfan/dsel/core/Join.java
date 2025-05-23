package com.moneyfan.dsel.core;

import java.util.function.Function;

/**
 * Represents a 2-ary tuple, the fundamental building block of the DSEL.
 * Emphasizes compositional purity and simplicity.
 *
 * @param <F> The type of the first element.
 * @param <S> The type of the second element.
 */
public interface Join<F, S> {

    /**
     * Gets the first element of this Join.
     *
     * @return The first element.
     */
    F first();

    /**
     * Gets the second element of this Join.
     *
     * @return The second element.
     */
    S second();

    /**
     * Factory method for creating a Join instance.
     *
     * @param f The first element.
     * @param s The second element.
     * @param <F> Type of the first element.
     * @param <S> Type of the second element.
     * @return A new Join instance.
     */
    static <F, S> Join<F, S> of(F f, S s) {
        return new JoinImpl<>(f, s);
    }

    /**
     * Applies a function to the first element of this Join.
     *
     * @param func The function to apply to the first element.
     * @param <R>  The type of the result of the function.
     * @return A new Join with the transformed first element and the original second element.
     */
    <R> Join<R, S> mapFirst(Function<? super F, ? extends R> func);

    /**
     * Applies a function to the second element of this Join.
     *
     * @param func The function to apply to the second element.
     * @param <R>  The type of the result of the function.
     * @return A new Join with the original first element and the transformed second element.
     */
    <R> Join<F, R> mapSecond(Function<? super S, ? extends R> func);

    /**
     * Swaps the elements of this Join.
     *
     * @return A new Join with the elements swapped.
     */
    Join<S, F> swap();

    /**
     * Applies a bifunction to both elements of this Join.
     *
     * @param fMap The function to apply to the first element.
     * @param sMap The function to apply to the second element.
     * @param <R1> The type of the first element in the new Join.
     * @param <R2> The type of the second element in the new Join.
     * @return A new Join with transformed elements.
     */
    <R1, R2> Join<R1, R2> bimap(Function<? super F, ? extends R1> fMap, Function<? super S, ? extends R2> sMap);
}

/**
 * Implementation of the Join interface.
 */
class JoinImpl<F, S> implements Join<F, S> {
    private final F first;
    private final S second;

    public JoinImpl(F first, S second) {
        this.first = first;
        this.second = second;
    }

    @Override
    public F first() {
        return first;
    }

    @Override
    public S second() {
        return second;
    }

    @Override
    public <R> Join<R, S> mapFirst(Function<? super F, ? extends R> func) {
        return new JoinImpl<>(func.apply(first), second);
    }

    @Override
    public <R> Join<F, R> mapSecond(Function<? super S, ? extends R> func) {
        return new JoinImpl<>(first, func.apply(second));
    }

    @Override
    public Join<S, F> swap() {
        return new JoinImpl<>(second, first);
    }

    @Override
    public <R1, R2> Join<R1, R2> bimap(Function<? super F, ? extends R1> fMap, Function<? super S, ? extends R2> sMap) {
        return new JoinImpl<>(fMap.apply(first), sMap.apply(second));
    }

    @Override
    public String toString() {
        return "(" + first + ", " + second + ")";
    }
}