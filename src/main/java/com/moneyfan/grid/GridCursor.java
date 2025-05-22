package com.moneyfan.grid;

import java.util.Collections;
import java.util.List;
import com.moneyfan.core.Scalar;
import java.util.stream.IntStream;

/**
 * Represents a 2D grid as vector-of-rows.
 */
public record GridCursor(Vect0r<RowVec> rows) {

    public int rowCount() {
        return rows.size();
    }

    public int columnCount() {
        return rowCount() == 0 ? 0 : rows.get(0).columnCount();
    }

    public RowVec getRow(int index) {
        return rows.get(index);
    }

    /**
     * Returns list of Scalar metadata for columns based on first row's cells.
     * If grid is empty, returns empty list.
     */
    public List<Scalar> getScalars() {
        if (rowCount() == 0) return Collections.emptyList();
        RowVec first = rows.get(0);
        return IntStream.range(0, first.columnCount())
                .mapToObj(first::get)
                .map(c -> c.meta().scalar())
                .toList();
    }

    public GridCursor select(String... columnNames) {
        if (rowCount()==0) return this;
        List<Scalar> scalars = getScalars();
        int[] indices = java.util.Arrays.stream(columnNames)
                .mapToInt(name -> {
                    for(int i=0;i<scalars.size();i++) if(scalars.get(i).name().equals(name)) return i;
                    throw new IllegalArgumentException("Unknown column " + name);
                }).toArray();
        Vect0r<RowVec> newRows = Vect0r.of(rowCount(), rowIdx -> {
            RowVec original = getRow(rowIdx);
            Vect0r<Cell> newCells = Vect0r.of(indices.length, colIdx -> original.get(indices[colIdx]));
            return new RowVec(newCells);
        });
        return new GridCursor(newRows);
    }

    public GridCursor mapColumn(String columnName, com.moneyfan.core.IOMemento newType, java.util.function.Function<Object,Object> transform) {
        if (rowCount()==0) return this;
        List<Scalar> scalars = getScalars();
        int colIndex=-1;
        for(int i=0;i<scalars.size();i++) if(scalars.get(i).name().equals(columnName)) { colIndex=i; break; }
        if(colIndex==-1) throw new IllegalArgumentException("Unknown column " + columnName);
        final int idx=colIndex;
        Scalar newScalar = com.moneyfan.core.Scalar.of(newType, columnName);
        Vect0r<RowVec> newRows = Vect0r.of(rowCount(), rowIdx -> {
            RowVec orig = getRow(rowIdx);
            Vect0r<Cell> newCells = Vect0r.of(columnCount(), col -> {
                Cell originalCell = orig.get(col);
                if(col==idx) {
                    Object newVal = transform.apply(originalCell.value());
                    com.moneyfan.core.CellMeta meta = new com.moneyfan.core.CellMeta(() -> newScalar);
                    return new Cell(newVal, meta);
                } else {
                    return originalCell;
                }
            });
            return new RowVec(newCells);
        });
        return new GridCursor(newRows);
    }

    public GridCursor filter(java.util.function.Predicate<RowVec> predicate) {
        java.util.List<Integer> matching = new java.util.ArrayList<>();
        for(int i=0;i<rowCount();i++) {
            RowVec row = getRow(i);
            if(predicate.test(row)) matching.add(i);
        }
        Vect0r<RowVec> newRows = Vect0r.of(matching.size(), idx -> getRow(matching.get(idx)));
        return new GridCursor(newRows);
    }
}