package com.vsiwest.bikeshed.dsel;

// This DSEL-specific Join can be used to add DSEL-specific methods or context
// without polluting the core Join implementation.
// For instance, operator-like methods or type aliases can live here.
// In Java, this would mostly be for clarity and namespacing, as true type aliases
// or operator overloading are not directly supported.

// Javadoc for type alias clarity (no actual Java `typealias`)
/**
 * @typedef Join<F, S> = com.example.bikeshed.core.Join<F, S>
 * This is a DSEL-specific `Join` providing additional convenience methods
 * and adhering to the `bbcursive` DSEL's philosophy.
 * It is effectively an alias for {@link com.example.bikeshed.core.Join}.
 */
public class Join<F, S> extends com.example.bikeshed.core.Join<F, S> {

    protected Join(F first, S second) {
        super(first, second);
    }

    // DSEL-specific factory method, could delegate to core.Join.of
    public static <F, S> Join<F, S> of(F first, S second) {
        return new Join<>(first, second);
    }

    /**
     * Operator-like method: `plus`. Conceptually allows `join1 + join2` if their types are compatible.
     * This is a placeholder; actual implementation depends on semantics (e.g., element-wise sum if numbers).
     *
     * @param other The other Join to combine with.
     * @param <F_O> Type of the first element of the other Join.
     * @param <S_O> Type of the second element of the other Join.
     * @return A new Join representing the combined result.
     */
    public <F_O, S_O> Join<Object, Object> plus(Join<F_O, S_O> other) {
        // Example: If F and S are numbers, this could be sum.
        // For general types, it could be a Join of Joins, or concatenation.
        // This is a design choice. For simplicity, let's make it a Join of the original Joins.
        return D.jn(this, other);
    }

    /**
     * Operator-like method: `div`. Conceptually allows `join / someValue`.
     * This is a placeholder; actual implementation depends on semantics (e.g., element-wise division if numbers).
     *
     * @param divisor The value to divide by.
     * @return A new Join with elements divided.
     */
    public Join<Double, Double> div(double divisor) {
        // This implementation assumes F and S are convertible to Double.
        // In a real DSEL, this would need type checking or specialized types.
        return D.jn(
                (this.getFirst() instanceof Number) ? ((Number) this.getFirst()).doubleValue() / divisor : Double.NaN,
                (this.getSecond() instanceof Number) ? ((Number) this.getSecond()).doubleValue() / divisor : Double.NaN
        );
    }

    /**
     * Provides concise access to the first element (glyph `a`).
     * @return The first element.
     */
    public F α() { return getFirst(); } // Using alpha as a glyph for "first" or "value" from a tuple

    /**
     * Provides concise access to the second element (glyph `b`).
     * @return The second element.
     */
    public S β() { return getSecond(); } // Using beta as a glyph for "second" or "metadata" from a tuple

    // Consider adding more such methods if they represent common DSEL operations.
}
