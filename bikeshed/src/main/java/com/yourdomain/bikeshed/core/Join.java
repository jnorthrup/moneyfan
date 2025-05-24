package com.yourdomain.bikeshed.core;

import org.jetbrains.annotations.NotNull;

import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * The foundational immutable 2-tuple primitive for the DSEL.
 * All operations on Join or its derived types must produce new instances.
 *
 * @param <F> The type of the first element.
 * @param <S> The type of the second element.
 */
public interface Join<F, S> {

    /**
     * Returns the first element of this Join.
     * @return The first element.
     */
    F fst();

    /**
     * Returns the second element of this Join.
     * @return The second element.
     */
    S snd();

    /**
     * Factory method to create a new Join instance.
     *
     * @param f The first element.
     * @param s The second element.
     * @param <F> The type of the first element.
     * @param <S> The type of the second element.
     * @return A new immutable Join instance.
     */
    static <F, S> @NotNull Join<F, S> of(F f, S s) {
        return new ImmutableJoin<>(f, s);
    }

    /**
     * Applies a function to the first element, producing a new Join instance.
     *
     * @param mapper The function to apply to the first element.
     * @param <R> The new type of the first element.
     * @return A new Join instance with the transformed first element.
     */
    default <R> @NotNull Join<R, S> mapFst(@NotNull Function<F, R> mapper) {
        return Join.of(mapper.apply(fst()), snd());
    }

    /**
     * Applies a function to the second element, producing a new Join instance.
     *
     * @param mapper The function to apply to the second element.
     * @param <R> The new type of the second element.
     * @return A new Join instance with the transformed second element.
     */
    default <R> @NotNull Join<F, R> mapSnd(@NotNull Function<S, R> mapper) {
        return Join.of(fst(), mapper.apply(snd()));
    }

    /**
     * Applies two functions to both elements, producing a new Join instance.
     *
     * @param fstMapper The function to apply to the first element.
     * @param sndMapper The function to apply to the second element.
     * @param <R1> The new type of the first element.
     * @param <R2> The new type of the second element.
     * @return A new Join instance with both elements transformed.
     */
    default <R1, R2> @NotNull Join<R1, R2> mapBoth(@NotNull Function<F, R1> fstMapper, @NotNull Function<S, R2> sndMapper) {
        return Join.of(fstMapper.apply(fst()), sndMapper.apply(snd()));
    }

    /**
     * Swaps the positions of the two elements, producing a new Join instance.
     *
     * @return A new Join instance with elements swapped.
     */
    default @NotNull Join<S, F> swap() {
        return Join.of(snd(), fst());
    }

    /**
     * Applies a function to both elements to produce a single result.
     * @param mapper The function to combine both elements.
     * @param <R> The type of the combined result.
     * @return The result of applying the mapper to both elements.
     */
    default <R> @NotNull R combine(@NotNull BiFunction<F, S, R> mapper) {
        return mapper.apply(fst(), snd());
    }

    // Inner class for the immutable implementation
    final class ImmutableJoin<F, S> implements Join<F, S> {
        private final F fst;
        private final S snd;

        private ImmutableJoin(F fst, S snd) {
            this.fst = fst;
            this.snd = snd;
        }

        @Override
        public F fst() {
            return fst;
        }

        @Override
        public S snd() {
            return snd;
        }

        @Override
        public String toString() {
            return "(" + fst + ", " + snd + ")";
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            ImmutableJoin<?, ?> that = (ImmutableJoin<?, ?>) o;
            return Objects.equals(fst, that.fst) && Objects.equals(snd, that.snd);
        }

        @Override
        public int hashCode() {
            return Objects.hash(fst, snd);
        }
    }
}
