package com.moneyfan.core;

import java.util.function.Supplier;

/**
 * Record for cell metadata provider.
 */
public record CellMeta(Supplier<Scalar> provider) {
    
    public static CellMeta of(Scalar scalar) {
        return new CellMeta(() -> scalar);
    }
    
    public Scalar getScalar() {
        return provider.get();
    }
}