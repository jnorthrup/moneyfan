package com.vsiwest.bikeshed.tuple;

import org.jetbrains.annotations.NotNull;
import java.util.Objects;

/**
 * Represents a pair of two elements of the same type.
 * This is a specialized {@link Join} where both elements have the same type.
 *
 * @param <T> The type of both elements.
 */
public interface Twin<T> extends Join<T, T> {

    /**
     * Factory method to create a new Twin instance.
     *
     * @param first The first element.
     * @param second The second element.
     * @param <T> The type of elements.
     * @return A new Twin instance.
     */
    static <T> @NotNull Twin<T> of(T first, T second) {
        return new ImmutableTwin<>(first, second);
    }

    /**
     * Returns the first element.
     * @return The first element.
     */
    @Override
    T fst();

    /**
     * Returns the second element.
     * @return The second element.
     */
    @Override
    T snd();

    /**
     * Immutable implementation of the Twin interface.
     */
    final class ImmutableTwin<T> extends Join.ImmutableJoin<T, T> implements Twin<T> {
        private ImmutableTwin(T first, T second) {
            super(first, second);
        }

        @Override
        public T fst() {
            return first();
        }

        @Override
        public T snd() {
            return second();
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            ImmutableTwin<?> that = (ImmutableTwin<?>) o;
            return Objects.equals(first(), that.first()) &&
                   Objects.equals(second(), that.second());
        }

        @Override
        public int hashCode() {
            return Objects.hash(first(), second());
        }

        @Override
        public String toString() {
            return "(" + first() + ", " + second() + ")";
        }
    }
}
