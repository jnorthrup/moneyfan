package com.moneyfan.dsl.core;

/**
 * MONEYFAN DESIGN PRINCIPLE:
 * <p>
 * This system is fundamentally built upon the {@code JPair<F,S>} 2-ary tuple as its atomic unit.
 * All components MUST leverage this pattern to enable:
 * <ol>
 *     <li>Strong identity formation through paired relationships</li>
 *     <li>Type-driven dispatch for specialized implementations</li>
 *     <li>Uniform composition patterns while preserving type safety</li>
 *     <li>Maximum pointer hoisting for JIT optimisation</li>
 * </ol>
 * Factory methods SHOULD examine pair types to select optimal implementations.
 */
public record JPair<F, S>(F first, S second) {

    /**
     * Static factory that improves type inference over the canonical constructor.
     */
    public static <F, S> JPair<F, S> of(F first, S second) {
        return new JPair<>(first, second);
    }

    /**
     * Map the first component, returning a new JPair.
     */
    public <R> JPair<R, S> mapFirst(java.util.function.Function<? super F, ? extends R> mapper) {
        return new JPair<>(mapper.apply(first), second);
    }

    /**
     * Map the second component, returning a new JPair.
     */
    public <R> JPair<F, R> mapSecond(java.util.function.Function<? super S, ? extends R> mapper) {
        return new JPair<>(first, mapper.apply(second));
    }
}