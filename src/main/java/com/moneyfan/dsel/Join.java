package com.moneyfan.dsel;

public interface Join<F, S> {
    F first();

    S second();

    /**
     * Static factory method to create an immutable Join instance.
     * This provides a simple, insulated entry point without exposing the core class.
     */
    static <F, S> Join<F, S> of(F first, S second) {
        return new jrec<>(first, second);  // Delegates to a simple implementation
    }

    /**
     * Static factory method to create an immutable Join instance.
     * This provides a simple, insulated entry point without exposing the core class.
     */
    record jrec<F, S>(F f, S s) implements Join<F, S> {
        @Override
        public F first() {
            return f;
        }

        @Override
        public S second() {
            return s;
        }
    }

    /**
     * just in case those records need a benchmark
     */
    static <F, S> Join<F, S> lite(F f, S s) {
        return new Join<>() {
            @Override
            public F first() {
                return f;
            }

            public S second() {
                return s;
            }
        };
    }
}