package com.moneyfan.common;

// A RowVec (Row Vector) represents a single row in a tabular data structure.
// It is a Series of Cells.
public interface RowVec extends Series<Cell> {
    // Row-specific operations can be added here.
    // For example, accessing cells by column name/index if metadata provides schema.
}
