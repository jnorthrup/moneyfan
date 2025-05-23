package com.moneyfan.common;

/**
 * A simple, immutable 2-ary tuple implementation using a Java Record.
 * This serves as the default concrete class for the {@link Join} interface.
 * Records automatically provide a canonical constructor, component accessor methods
 * (named after the components, e.g., {@code first()}, {@code second()}),
 * {@code equals()}, {@code hashCode()}, and {@code toString()}.
 *
 * @param <F> Type of the first element.
 * @param <S> Type of the second element.
 * @param first The first element of the tuple.
 * @param second The second element of the tuple.
 */
public record SimpleJoin<F, S>(F first, S second) implements Join<F, S> {

    // The record implicitly has:
    // - public SimpleJoin(F first, S second) { ... }
    // - public F first() { return this.first; }
    // - public S second() { return this.second; }
    // To align with the Join interface's getFirst()/getSecond() method names:
    @Override
    public F getFirst() {
        return first; // Delegates to the record's 'first' component accessor.
    }
    @Override
    public S getSecond() {
        return second; // Delegates to the record's 'second' component accessor.
    }
}
