package com.moneyfan.grid;

import com.moneyfan.core.CellMeta;
import java.util.Objects;

/**
 * A data cell consisting of a {@code value} and accompanying metadata supplier.
 * The value type should conform to the {@link com.moneyfan.core.IOMemento java type} declared by
 * the {@link CellMeta}'s {@link com.moneyfan.core.Scalar}.
 */
public record Cell(Object value, CellMeta meta) {

    public Cell {
        Objects.requireNonNull(meta, "meta");
    }
}