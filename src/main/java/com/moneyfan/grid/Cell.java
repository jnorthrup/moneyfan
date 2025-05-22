package com.moneyfan.grid;

import com.moneyfan.core.CellMeta;

/**
 * Cell stores a value along with metadata describing its column.
 */
public record Cell(Object value, CellMeta meta) {
}