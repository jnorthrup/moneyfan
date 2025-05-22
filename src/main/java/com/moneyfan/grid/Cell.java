package com.moneyfan.grid;

import com.moneyfan.core.CellMeta;

/**
 * Record for cell value and its metadata.
 */
public record Cell(Object value, CellMeta meta) {
    
    public static Cell of(Object value, CellMeta meta) {
        return new Cell(value, meta);
    }
}