package com.moneyfan.common;

/**
 * Abstract interface for a 2-ary tuple, immutable.
 * This is the core immutable data structure for the DSEL, kept simple to minimize conceptual overhead.
 * Static factories insulate users from the implementation details.
 *
 * @param <F> Type of the first element
 * @param <S> Type of the second element
 */
public interface Join<F, S> {
    F getFirst();
    S getSecond();

    /**
     * Static factory method to create an immutable Join instance.
     * This provides a simple, insulated entry point without exposing the core class.
     */
    static <F, S> Join<F, S> of(F first, S second) {
        return new ImmutableJoin<>(first, second);  // Delegates to a simple implementation
    }
}
