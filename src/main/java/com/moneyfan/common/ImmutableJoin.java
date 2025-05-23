package com.moneyfan.common;

import java.util.Objects;

/**
 * An immutable custom implementation of Join.
 * This class had errors related to missing `getSecond` and using `first()`/`second()`
 * instead of `getFirst()`/`getSecond()`.
 */
public final class ImmutableJoin<F, S> implements Join<F, S> {
    private final F firstValue;
    private final S secondValue;

    public ImmutableJoin(F firstValue, S secondValue) {
        this.firstValue = Objects.requireNonNull(firstValue, "firstValue cannot be null");
        this.secondValue = Objects.requireNonNull(secondValue, "secondValue cannot be null");
    }

    @Override
    public F getFirst() {
        return firstValue;
    }

    @Override
    public S getSecond() {
        return secondValue;
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
        return Objects.hash(firstValue, secondValue);
    }

    @Override
    public String toString() {
        return "ImmutableJoin(" + firstValue + ", " + secondValue + ")";
    }
}
