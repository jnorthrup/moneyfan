package com.example.bikeshed.core;

import com.example.bikeshed.dsel.D;

import java.util.Objects;
import java.util.function.Function;

/**
 * The foundational immutable 2-tuple.
 * All operations on Join or its derived types must be compositional,
 * meaning they produce *new* instances rather than modifying existing ones.
 * This immutability promotes predictable behavior, simplifies concurrency,
 * and enables powerful optimization techniques (e.g., memoization, lazy evaluation).
 *
 * This class serves as the core, immutable Pair/Join implementation.
 * The DSEL layer (`com.example.bikeshed.dsel.Join`) will extend or wrap this.
 *
 * @param <F> The type of the first element.
 * @param <S> The type of the second element.
 */
public class Join<F, S> {
    private final F first;
    private final S second;

    protected Join(F first, S second) {
        this.first = first;
        this.second = second;
    }

    /**
     * Factory method for creating a Join instance.
     * This method is typically accessed via the DSEL utility enum (e.g., D.jn(f, s)).
     *
     * @param first  The first element.
     * @param second The second element.
     * @param <F>    Type of the first element.
     * @param <S>    Type of the second element.
     * @return A new immutable Join instance.
     */
    public static <F, S> Join<F, S> of(F first, S second) {
        return new Join<>(first, second);
    }

    public F getFirst() {
        return first;
    }

    public S getSecond() {
        return second;
    }

    /**
     * Type alias equivalent: `typealias Join<F, S> = Pair<F, S>`
     * In Java, this is represented by naming convention.
     * Getters for first and second elements using conventional names `a` and `b`.
     * These are often used as "operator overloading via convention" in the DSEL.
     */
    public F a() { return first; }
    public S b() { return second; }

    /**
     * Maps the first element of the Join to a new value.
     * Compositional: returns a new Join instance.
     *
     * @param mapper Function to apply to the first element.
     * @param <R>    New type of the first element.
     * @return A new Join instance with the transformed first element.
     */
    public <R> Join<R, S> mapFst(Function<? super F, ? extends R> mapper) {
        return D.jn(mapper.apply(this.first), this.second);
    }

    /**
     * Maps the second element of the Join to a new value.
     * Compositional: returns a new Join instance.
     *
     * @param mapper Function to apply to the second element.
     * @param <R>    New type of the second element.
     * @return A new Join instance with the transformed second element.
     */
    public <R> Join<F, R> mapSnd(Function<? super S, ? extends R> mapper) {
        return D.jn(this.first, mapper.apply(this.second));
    }

    /**
     * Maps both elements of the Join to new values.
     * Compositional: returns a new Join instance.
     *
     * @param mapperFst Function to apply to the first element.
     * @param mapperSnd Function to apply to the second element.
     * @param <R1>      New type of the first element.
     * @param <R2>      New type of the second element.
     * @return A new Join instance with both elements transformed.
     */
    public <R1, R2> Join<R1, R2> mapBoth(Function<? super F, ? extends R1> mapperFst,
                                         Function<? super S, ? extends R2> mapperSnd) {
        return D.jn(mapperFst.apply(this.first), mapperSnd.apply(this.second));
    }

    /**
     * Swaps the elements of the Join.
     * Compositional: returns a new Join instance.
     *
     * @return A new Join instance with elements swapped.
     */
    public Join<S, F> swap() {
        return D.jn(this.second, this.first);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Join<?, ?> join = (Join<?, ?>) o;
        return Objects.equals(first, join.first) &&
               Objects.equals(second, join.second);
    }

    @Override
    public int hashCode() {
        return Objects.hash(first, second);
    }

    @Override
    public String toString() {
        return "(" + first + ", " + second + ")";
    }
}
