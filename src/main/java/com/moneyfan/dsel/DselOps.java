package com.moneyfan.dsel;

import java.util.function.Function;

/**
 * Enum housing common operations or acting as singletons for logic.
 * This demonstrates the "enums for bags of code elements" concept.
 */
public enum DselOps {
    // Example: An operation that might be used in a groupBy or key extraction
    IDENTITY_KEY_EXTRACTOR {
        // This could be a Function<T, T> if T is the key itself
        public <T> Function<T, T> asKey() {
            return t -> t;
        };
    },

    // The original SWAP_JOIN is a good example of an operation on a Join
    // However, join.swap() is often more direct.
    // This enum form is useful if you want to pass the *operation itself* around.
    SWAP_JOIN {
        public <F, S> Function<Join<F, S>, Join<S, F>> asFunction() {
            return Join::swap;
        }

        // Direct application if preferred for some contexts
        public <F, S> Join<S, F> apply(Join<F, S> join) {
            return join.swap();
        }
    };

    /**
     * Enum constants can provide methods that return specific functional interfaces
     * or perform direct actions. This is more flexible than a single abstract method.
     */
    // Example of a more specific abstract method if all ops shared a very common signature:
    // public abstract <T, R> Function<T, R> getMapper();
}
