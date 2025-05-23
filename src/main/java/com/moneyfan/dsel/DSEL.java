package com.moneyfan.dsel;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

/**
 * DSEL (Domain Specific Embedded Language) entry point.
 * Provides factory methods for creating {@link DSEL_Cursor} instances.
 * Implemented as a singleton enum as per the design request.
 */
public enum DSEL {
    INSTANCE; // Singleton entry point

    public <F, S> DSEL_Cursor<F, S> from(List<Join<F, S>> list) {
        return new ListBackedCursor<>(list);
    }

    @SafeVarargs
    public final <F, S> DSEL_Cursor<F, S> of(Join<F, S>... joins) {
        return new ListBackedCursor<>(Arrays.asList(joins));
    }

    public <F, S> DSEL_Cursor<F, S> fromStream(Stream<Join<F, S>> stream) {
        return new ListBackedCursor<>(stream.toList());
    }
}
