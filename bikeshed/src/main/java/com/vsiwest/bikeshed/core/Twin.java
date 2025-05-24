package com.vsiwest.bikeshed.core;

import org.jetbrains.annotations.NotNull;

/**
 * A specialized {@link Join} for symmetric pairs, where both elements are of the same type.
 *
 * @param <T> The common type of both elements.
 */
public interface Twin<T> extends Join<T, T> {

    /**
     * Factory method to create a new Twin instance.
     *
     * @param fst The first element.
     * @param snd The second element.
     * @param <T> The type of elements.
     * @return A new immutable Twin instance.
     */
    static <T> @NotNull Twin<T> of(T fst, T snd) {
        return new ImmutableTwin<>(fst, snd);
    }

    // Inner class for the immutable implementation
    final class ImmutableTwin<T> extends Join.ImmutableJoin<T, T> implements Twin<T> {
        private ImmutableTwin(T fst, T snd) {
            super(fst, snd);
        }
    }
}
