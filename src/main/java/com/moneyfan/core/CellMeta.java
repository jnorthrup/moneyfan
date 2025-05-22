package com.moneyfan.core;

import java.util.Objects;
import java.util.function.Supplier;

/**
 * Provides lazy access to column Scalar metadata for a Cell.
 */
public record CellMeta(Supplier<Scalar> provider) {
    public CellMeta {
        Objects.requireNonNull(provider, "provider");
    }

    public Scalar scalar() {
        return provider.get();
    }
}