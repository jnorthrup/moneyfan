package com.moneyfan.common;

/**
 * A class that implements Join. If it's intended to be a simple pair,
 * consider if SimpleJoin (the record) is sufficient or if Cell has other semantics.
 * Assuming it's a basic Join implementation for now.
 * The errors indicated it had first() and second() methods.
 */
public final class Cell<F, S> implements Join<F, S> {
    private final F firstValue;
    private final S secondValue;

    public Cell(F firstValue, S secondValue) {
        this.firstValue = firstValue; // Example: key
        this.secondValue = secondValue; // Example: value
    }

    @Override
    public F getFirst() {
        return firstValue;
    }

    @Override
    public S getSecond() {
        return secondValue;
    }

    // equals, hashCode, toString would be good additions if not using a record
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Join)) return false;
        Join<?, ?> that = (Join<?, ?>) o;
        return this.getFirst().equals(that.getFirst()) && this.getSecond().equals(that.getSecond());
    }

    @Override
    public int hashCode() {
        return java.util.Objects.hash(getFirst(), getSecond());
    }

    @Override
    public String toString() {
        return "Cell(" + getFirst() + ", " + getSecond() + ")";
    }
}
