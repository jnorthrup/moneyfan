package com.moneyfan.core;

import java.util.function.Supplier;

/**
 * A lightweight wrapper that provides access to the {@link Scalar} metadata for a cell.
 * Supplier indirection allows sharing Scalar instances lazily across cells.
 */
public record CellMeta(Supplier<Scalar> provider) {
}