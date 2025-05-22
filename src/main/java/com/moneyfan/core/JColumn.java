package com.moneyfan.core;

import java.util.List;

/**
 * A class representing a column in a 2D grid, holding a list of values.
 * This immutable data structure is designed for high-performance grid operations.
 */
public final class JColumn<T> {
    private final int id;
    private final List<T> values;
    
    public JColumn(int id, List<T> values) {
        this.id = id;
        this.values = values;
    }
    
    public int getId() {
        return id;
    }
    
    public List<T> getValues() {
        return values;
    }
    
    /**
     * Returns the value at the specified row index.
     * @param rowIndex the index of the row
     * @return the value at the specified row index
     */
    public T getValue(int rowIndex) {
        return values.get(rowIndex);
    }
    
    /**
     * Returns the size of the column (number of rows).
     * @return the number of values in the column
     */
    public int size() {
        return values.size();
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        JColumn<?> jColumn = (JColumn<?>) o;
        return id == jColumn.id && values.equals(jColumn.values);
    }
    
    @Override
    public int hashCode() {
        int result = 17;
        result = 31 * result + id;
        result = 31 * result + values.hashCode();
        return result;
    }
}