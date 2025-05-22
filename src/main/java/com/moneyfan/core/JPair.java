package com.moneyfan.core;

/**
 * A class representing a pair of integer coordinates (row, column) for 2D grid manipulation.
 * This immutable data structure is optimized for high-performance operations.
 */
public final class JPair {
    private final int row;
    private final int col;
    
    public JPair(int row, int col) {
        this.row = row;
        this.col = col;
    }
    
    public int row() {
        return row;
    }
    
    public int col() {
        return col;
    }
    
    /**
     * Creates a new JPair with translated coordinates.
     * @param rowOffset the offset to add to the row
     * @param colOffset the offset to add to the column
     * @return a new JPair instance with the translated coordinates
     */
    public JPair translate(int rowOffset, int colOffset) {
        return new JPair(this.row + rowOffset, this.col + colOffset);
    }
    
    @Override
    public String toString() {
        return "(" + row + ", " + col + ")";
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        JPair jPair = (JPair) o;
        return row == jPair.row && col == jPair.col;
    }
    
    @Override
    public int hashCode() {
        int result = 17;
        result = 31 * result + row;
        result = 31 * result + col;
        return result;
    }
}