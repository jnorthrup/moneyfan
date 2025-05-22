package com.moneyfan.grid;

import com.moneyfan.core.CellMeta;

/**
 * Immutable cell wrapper holding the raw value and its metadata.
 */
public record Cell(Object value, CellMeta meta) {
}