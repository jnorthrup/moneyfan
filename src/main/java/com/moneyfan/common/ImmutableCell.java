package com.moneyfan.common;

import java.util.Objects;

/**
 * An immutable cell, implementing Join.
 * The error "not abstract and does not override abstract method getSecond()"
 * means it needs to properly implement getSecond() from the Join interface.
 */
public final class ImmutableCell<F, S> implements Join<F, S> {
    private final F first;
    private final S second;

    public ImmutableCell(F first, S second) {
        this.first = Objects.requireNonNull(first, "first cannot be null");
        this.second = Objects.requireNonNull(second, "second cannot be null");
    }

    @Override
    public F getFirst() {
        return first;
    }

    @Override
    public S getSecond() {
        return second;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Join)) return false;
        Join<?, ?> that = (Join<?, ?>) o;
        return Objects.equals(getFirst(), that.getFirst()) &&
               Objects.equals(getSecond(), that.getSecond());
    }

    @Override
    public int hashCode() {
        return Objects.hash(first, second);
    }

    @Override
    public String toString() {
        return "ImmutableCell(" + first + ", " + second + ")";
    }
}
