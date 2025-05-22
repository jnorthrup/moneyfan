package com.moneyfan.grid;

import com.moneyfan.core.Scalar;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;

/**
 * Record representing the 2D grid as a vector of row vectors.
 */
public record GridCursor(Vect0r<RowVec> rows) {
    
    public static GridCursor of(Vect0r<RowVec> rows) {
        return new GridCursor(rows);
    }
    
    public int rowCount() {
        return rows.size();
    }
    
    public int columnCount() {
        return rows.size() > 0 ? rows.get(0).cells().size() : 0;
    }
    
    public RowVec getRow(int rowIndex) {
        return rows.get(rowIndex);
    }
    
    public List<Scalar> getScalars() {
        if (rowCount() == 0) return List.of();
        
        RowVec firstRow = getRow(0);
        List<Scalar> scalars = new ArrayList<>(firstRow.cells().size());
        
        for (int i = 0; i < firstRow.cells().size(); i++) {
            scalars.add(firstRow.getCell(i).meta().getScalar());
        }
        
        return scalars;
    }
    
    // Basic DSL operations
    
    public GridCursor select(String... columnNames) {
        List<Integer> indices = new ArrayList<>();
        List<Scalar> scalars = getScalars();
        
        for (String name : columnNames) {
            for (int i = 0; i < scalars.size(); i++) {
                if (scalars.get(i).name().equals(name)) {
                    indices.add(i);
                    break;
                }
            }
        }
        
        return new GridCursor(Vect0r.of(rowCount(), rowIdx -> {
            RowVec originalRow = getRow(rowIdx);
            Vect0r<Cell> selectedCells = Vect0r.of(indices.size(), colIdx -> 
                originalRow.getCell(indices.get(colIdx))
            );
            return new RowVec(selectedCells);
        }));
    }
    
    public GridCursor filter(Predicate<RowVec> predicate) {
        List<Integer> matchingRows = new ArrayList<>();
        
        for (int i = 0; i < rowCount(); i++) {
            if (predicate.test(getRow(i))) {
                matchingRows.add(i);
            }
        }
        
        return new GridCursor(Vect0r.of(matchingRows.size(), idx -> 
            getRow(matchingRows.get(idx))
        ));
    }
}