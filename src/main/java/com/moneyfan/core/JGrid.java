package com.moneyfan.core;

import java.util.List;
import java.util.ArrayList;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * A class representing a 2D grid for high-performance data manipulation.
 * This structure uses JColumn and JPair for data organization and access.
 */
public final class JGrid<T> {
    private final int rows;
    private final int cols;
    private final List<JColumn<T>> columns;
    
    public JGrid(int rows, int cols, List<JColumn<T>> columns) {
        this.rows = rows;
        this.cols = cols;
        this.columns = columns;
    }
    
    public int getRows() {
        return rows;
    }
    
    public int getCols() {
        return cols;
    }
    
    public List<JColumn<T>> getColumns() {
        return columns;
    }
    
    /**
     * Retrieves the value at the specified coordinates.
     * @param coord the JPair representing row and column coordinates
     * @return the value at the specified position
     */
    public T getValue(JPair coord) {
        return columns.get(coord.col()).getValue(coord.row());
    }
    
    /**
     * Retrieves the column at the specified index.
     * @param colIndex the index of the column
     * @return the JColumn at the specified index
     */
    public JColumn<T> getColumn(int colIndex) {
        return columns.get(colIndex);
    }
    
    /**
     * Calculates the sum of all values in a specified column, assuming the values are numeric.
     * @param colIndex the index of the column to sum
     * @return the sum of the values in the column as a double
     */
    public double sumColumn(int colIndex) {
        JColumn<T> column = columns.get(colIndex);
        double sum = 0.0;
        for (int i = 0; i < rows; i++) {
            T value = column.getValue(i);
            if (value instanceof Number) {
                sum += ((Number) value).doubleValue();
            }
        }
        return sum;
    }

    /**
     * Applies a transformation function to each value in the grid, returning a new JGrid with transformed values.
     * @param transformer a function to transform each value
     * @return a new JGrid with transformed values
     */
    @SuppressWarnings("unchecked")
    public <R> JGrid<R> transform(Function<T, R> transformer) {
        ArrayList<JColumn<R>> newColumns = new ArrayList<>();
        for (JColumn<T> column : columns) {
            ArrayList<R> transformedValues = new ArrayList<>();
            for (int i = 0; i < rows; i++) {
                transformedValues.add(transformer.apply(column.getValue(i)));
            }
            newColumns.add(createColumnFromList(transformedValues));
        }
        return new JGrid<>(rows, cols, newColumns);
    }

    /**
     * Filters rows based on a predicate applied to values in a specific column, returning a new JGrid.
     * @param colIndex the index of the column to apply the predicate on
     * @param predicate the condition to filter rows
     * @return a new JGrid containing only the rows that match the predicate
     */
    public JGrid<T> filterByColumn(int colIndex, Predicate<T> predicate) {
        ArrayList<Integer> matchingRows = new ArrayList<>();
        JColumn<T> column = columns.get(colIndex);
        for (int i = 0; i < rows; i++) {
            if (predicate.test(column.getValue(i))) {
                matchingRows.add(i);
            }
        }
        
        ArrayList<JColumn<T>> filteredColumns = new ArrayList<>();
        for (JColumn<T> col : columns) {
            ArrayList<T> filteredValues = new ArrayList<>();
            for (int row : matchingRows) {
                filteredValues.add(col.getValue(row));
            }
            filteredColumns.add(createColumnFromList(filteredValues));
        }
        return new JGrid<>(matchingRows.size(), cols, filteredColumns);
    }

    /**
     * Computes the average of values in a specified column, assuming the values are numeric.
     * @param colIndex the index of the column to average
     * @return the average of the values in the column as a double
     */
    public double averageColumn(int colIndex) {
        if (rows == 0) return 0.0;
        return sumColumn(colIndex) / rows;
    }
    
    /**
     * Helper method to create a JColumn from a list of values.
     * This assumes JColumn has a constructor or factory method for List.
     * Replace with actual implementation based on JColumn's API.
     */
    private <R> JColumn<R> createColumnFromList(List<R> values) {
        // Using 0 as a default id for new columns; adjust as needed based on application logic
        return new JColumn<>(0, values);
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        JGrid<?> jGrid = (JGrid<?>) o;
        return rows == jGrid.rows && cols == jGrid.cols && columns.equals(jGrid.columns);
    }
    
    @Override
    public int hashCode() {
        int result = 17;
        result = 31 * result + rows;
        result = 31 * result + cols;
        result = 31 * result + columns.hashCode();
        return result;
    }
}