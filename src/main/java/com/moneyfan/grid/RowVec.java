package com.moneyfan.grid;

/**
 * Represents a row as a vector of cells.
 */
public record RowVec(Vect0r<Cell> cells) {

    public int columnCount() {
        return cells.size();
    }

    public Cell get(int column) {
        return cells.get(column);
    }
}