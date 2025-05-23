package com.moneyfan.dsel;

import java.util.function.Function;
import java.util.function.Predicate;

/**
 * Placeholder for utility functions or enum-based operations if needed later.
 * For now, most operations are directly on DSEL_Cursor or Join.
 * The CommonOps enum provides pre-defined predicates and functions.
 *
 * This enum demonstrates the "enum as a bag of code elements" for singleton objects
 * or collections of related static utility methods, if desired.
 */
public enum OpUtils {
    INSTANCE; // Singleton instance if this enum were to hold state or be a true singleton service

    // Example of a higher-order function that could be part of OpUtils
    public static <T, U, R> Function<Join<T, U>, R> applyToFirst(Function<T, R> func) {
        return join -> func.apply(join.first());
    }
}
