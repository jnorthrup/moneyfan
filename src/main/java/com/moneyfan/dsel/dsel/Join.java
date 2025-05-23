package com.moneyfan.dsel.dsel;

import java.util.function.Function;

/**
 * The core 2-ary tuple record for the DSEL.
 * @param <F> type of the first element
 * @param <S> type of the second element
 */
public record Join<F, S>(F first, S second) {
    /**
     * Maps the first element of this Join to a new value, creating a new Join.
     * @param <F2> the type of the new first element
     * @param mapper a function to apply to the first element
     * @return a new Join with the transformed first element and original second element
     */
    public <F2> Join<F2, S> mapFirst(Function<? super F, ? extends F2> mapper) {
        return new Join<>(mapper.apply(first), second);
    }

    /**
     * Maps the second element of this Join to a new value, creating a new Join.
     * @param <S2> the type of the new second element
     * @param mapper a function to apply to the second element
     * @return a new Join with the original first element and transformed second element
     */
    public <S2> Join<F, S2> mapSecond(Function<? super S, ? extends S2> mapper) {
        return new Join<>(first, mapper.apply(second));
    }

    /**
     * Maps both elements of this Join to new values, creating a new Join.
     * @param <F2> the type of the new first element
     * @param <S2> the type of the new second element
     * @param firstMapper a function to apply to the first element
     * @param secondMapper a function to apply to the second element
     * @return a new Join with transformed first and second elements
     */
    public <F2, S2> Join<F2, S2> map(Function<? super F, ? extends F2> firstMapper, Function<? super S, ? extends S2> secondMapper) {
        return new Join<>(firstMapper.apply(first), secondMapper.apply(second));
    }

    @Override
    public String toString() {
        return "Join[first=" + first + ", second=" + second + "]";
    }
}
