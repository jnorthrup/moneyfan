package com.moneyfan.dsel.core.enums;

import com.moneyfan.dsel.core.Join;

import java.util.function.Function;
import java.util.function.UnaryOperator;

/**
 * Enum for DSEL operations, singletons, and concise syntax providers.
 * This demonstrates encapsulating DSEL elements within enums.
 */
public enum Ops {
    /**
     * Singleton representing a Nil or empty value, potentially typed via context.
     * For example, Join.jn(Ops.NIL, Ops.NIL) could be an empty Join.
     */
    NIL; // Example singleton

    /**
     * Concise factory for Join, an alternative to Join.jn if shorter syntax is desired
     * via static import of this enum's members.
     * e.g. import static com.moneyfan.dsel.core.enums.Ops.*; j(1,2)
     */
    public static <F, S> Join<F, S> j(F first, S second) {
        return Join.jn(first, second);
    }

    // --- Unary Operators on Joins (examples) ---

    /**
     * Example of a named unary operator for Joins.
     * This could be part of a library of standard operations.
     * Takes a Join and returns its swapped version.
     */
    public static <F, S> UnaryOperator<Join<F, S>> swap() {
        return Join::swap;
    }

    // --- Series related operations (conceptual) ---
    // Series: Join<Integer, Function<Integer, T>>
    // Operations like `get(index)` for a series would be methods on a Series wrapper or static methods here.
}
