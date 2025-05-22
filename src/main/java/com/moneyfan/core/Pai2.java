package com.moneyfan.core;

import java.util.Objects;

/**
 * An immutable pair of {@code first} and {@code second} values.
 * <p>
 *     This is the spiritual successor to Kotlin's <em>Pai2</em> alias that was frequently used
 *     throughout the original columnar code-base.  In Java we model it as a {@link java.lang.Record}
 *     for conciseness and memory-layout benefits.
 * </p>
 * @param <F> first element type
 * @param <S> second element type
 */
public record Pai2<F, S>(F first, S second) {

    /**
     * Static factory method with type inference convenience.
     */
    public static <F, S> Pai2<F, S> of(F first, S second) {
        return new Pai2<>(first, second);
    }

    /**
     * Maps the first component with the supplied mapper leaving the second untouched.
     */
    public <NF> Pai2<NF, S> mapFirst(java.util.function.Function<? super F, ? extends NF> mapper) {
        Objects.requireNonNull(mapper, "mapper");
        return new Pai2<>(mapper.apply(first), second);
    }

    /**
     * Maps the second component with the supplied mapper leaving the first untouched.
     */
    public <NS> Pai2<F, NS> mapSecond(java.util.function.Function<? super S, ? extends NS> mapper) {
        Objects.requireNonNull(mapper, "mapper");
        return new Pai2<>(first, mapper.apply(second));
    }
}