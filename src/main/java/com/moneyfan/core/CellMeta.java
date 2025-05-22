package com.moneyfan.core;

import java.util.Objects;
import java.util.function.Supplier;

/**
 * Meta-data for a cell provided lazily via {@link Supplier}.  This allows delaying costly schema
 * creation until first access whilst remaining thread-safe thanks to record immutability.
 */
public record CellMeta(Supplier<Scalar> scalarProvider) {

    public CellMeta {
        Objects.requireNonNull(scalarProvider, "scalarProvider");
    }

    public Scalar scalar() {
        return scalarProvider.get();
    }
}