package com.moneyfan.grid;

/**
 * Record representing a row as a vector of cells.
 */
public record RowVec(Vect0r<Cell> cells) {
    
    public static RowVec of(Vect0r<Cell> cells) {
        return new RowVec(cells);
    }
    
    public Cell getCell(int index) {
        return cells.get(index);
    }
    
    public Object getValue(int index) {
        return getCell(index).value();
    }
}