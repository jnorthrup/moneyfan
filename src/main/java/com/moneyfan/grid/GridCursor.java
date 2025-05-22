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

    public GridCursor sortBy(String columnName, java.util.Comparator<Object> comparator) {
        if (rowCount()==0) return this;
        int tmpIdx = -1;
        List<Scalar> scalars = getScalars();
        for(int i=0;i<scalars.size();i++) if(scalars.get(i).name().equals(columnName)) { tmpIdx=i; break; }
        if(tmpIdx==-1) throw new IllegalArgumentException("Unknown column " + columnName);
        final int colIdx = tmpIdx;
        java.util.List<RowVec> copy = new java.util.ArrayList<>(rowCount());
        for(RowVec row : this.rows) copy.add(row);
        copy.sort((a,b) -> comparator.compare(a.get(colIdx).value(), b.get(colIdx).value()));
        return new GridCursor(Vect0r.fromList(copy));
    }

    public java.util.Map<Object, GridCursor> groupBy(String columnName) {
        if (rowCount()==0) return java.util.Collections.emptyMap();
        int tmpIdx2 = -1;
        List<Scalar> scalars = getScalars();
        for(int i=0;i<scalars.size();i++) if(scalars.get(i).name().equals(columnName)) { tmpIdx2=i; break; }
        if(tmpIdx2==-1) throw new IllegalArgumentException("Unknown column " + columnName);
        final int colIdx = tmpIdx2;
        java.util.Map<Object, java.util.List<RowVec>> map = new java.util.HashMap<>();
        for(RowVec row : this.rows) {
            Object key = row.get(colIdx).value();
            map.computeIfAbsent(key, k -> new java.util.ArrayList<>()).add(row);
        }
        java.util.Map<Object, GridCursor> result = new java.util.HashMap<>();
        map.forEach((k, v) -> result.put(k, new GridCursor(Vect0r.fromList(v))));
        return java.util.Collections.unmodifiableMap(result);
    }

    public GridCursor innerJoin(GridCursor other, String columnName) {
        if (rowCount()==0 || other.rowCount()==0) return new GridCursor(Vect0r.of(0, i->null));
        List<Scalar> leftScalars = getScalars();
        List<Scalar> rightScalars = other.getScalars();
        int leftIdx=-1,rightIdx=-1;
        for(int i=0;i<leftScalars.size();i++) if(leftScalars.get(i).name().equals(columnName)) { leftIdx=i; break; }
        for(int i=0;i<rightScalars.size();i++) if(rightScalars.get(i).name().equals(columnName)) { rightIdx=i; break; }
        if(leftIdx==-1||rightIdx==-1) throw new IllegalArgumentException("Join column missing");
        java.util.Map<Object, java.util.List<RowVec>> rightMap = new java.util.HashMap<>();
        for(RowVec r: other.rows) {
            Object key=r.get(rightIdx).value();
            rightMap.computeIfAbsent(key,k->new java.util.ArrayList<>()).add(r);
        }
        java.util.List<RowVec> outRows = new java.util.ArrayList<>();
        for(RowVec lrow: this.rows) {
            Object key = lrow.get(leftIdx).value();
            var matches = rightMap.get(key);
            if(matches!=null) {
                for(RowVec rrow: matches) {
                    // concatenate cells
                    java.util.List<Cell> combined = new java.util.ArrayList<>(lrow.columnCount()+rrow.columnCount());
                    for(int i=0;i<lrow.columnCount();i++) combined.add(lrow.get(i));
                    for(int j=0;j<rrow.columnCount();j++) combined.add(rrow.get(j));
                    outRows.add(new RowVec(Vect0r.fromList(combined)));
                }
            }
        }
        return new GridCursor(Vect0r.fromList(outRows));
    }

    public GridCursor sumBy(String groupByColumn, String targetColumn, String resultColumnName) {
        if (rowCount()==0) return this;
        List<Scalar> scalars = getScalars();
        int groupIdx=-1,targetIdx=-1;
        for(int i=0;i<scalars.size();i++) {
            if(scalars.get(i).name().equals(groupByColumn)) groupIdx=i;
            if(scalars.get(i).name().equals(targetColumn)) targetIdx=i;
        }
        if(groupIdx==-1||targetIdx==-1) throw new IllegalArgumentException("Missing columns");
        java.util.Map<Object, Double> sums = new java.util.HashMap<>();
        for(RowVec row: rows) {
            Object key = row.get(groupIdx).value();
            double val = ((Number) row.get(targetIdx).value()).doubleValue();
            sums.merge(key, val, Double::sum);
        }
        Scalar groupScalar = scalars.get(groupIdx);
        Scalar sumScalar = com.moneyfan.core.Scalar.of(com.moneyfan.core.IOMemento.IO_DOUBLE, resultColumnName);
        java.util.List<RowVec> outRows = new java.util.ArrayList<>(sums.size());
        sums.forEach((k,v)->{
            java.util.List<Cell> cells = java.util.List.of(
                    new Cell(k, new com.moneyfan.core.CellMeta(() -> groupScalar)),
                    new Cell(v, new com.moneyfan.core.CellMeta(() -> sumScalar))
            );
            outRows.add(new RowVec(Vect0r.fromList(cells)));
        });
        return new GridCursor(Vect0r.fromList(outRows));
    }

    public GridCursor leftOuterJoin(GridCursor right, String columnName) {
        if (rowCount()==0) return this;
        List<Scalar> leftScalars = getScalars();
        List<Scalar> rightScalars = right.getScalars();
        int leftIdx=-1,rightIdx=-1;
        for(int i=0;i<leftScalars.size();i++) if(leftScalars.get(i).name().equals(columnName)) { leftIdx=i; break; }
        for(int i=0;i<rightScalars.size();i++) if(rightScalars.get(i).name().equals(columnName)) { rightIdx=i; break; }
        if(leftIdx==-1||rightIdx==-1) throw new IllegalArgumentException("Join column missing");
        java.util.Map<Object, java.util.List<RowVec>> rightMap = new java.util.HashMap<>();
        for(RowVec r: right.rows) {
            Object key=r.get(rightIdx).value();
            rightMap.computeIfAbsent(key,k->new java.util.ArrayList<>()).add(r);
        }
        java.util.List<RowVec> outRows = new java.util.ArrayList<>();
        for(RowVec lrow: this.rows) {
            Object key = lrow.get(leftIdx).value();
            var matches = rightMap.get(key);
            if(matches!=null) {
                for(RowVec rrow: matches) {
                    java.util.List<Cell> combined = new java.util.ArrayList<>(lrow.columnCount()+rrow.columnCount());
                    for(int i=0;i<lrow.columnCount();i++) combined.add(lrow.get(i));
                    for(int j=0;j<rrow.columnCount();j++) combined.add(rrow.get(j));
                    outRows.add(new RowVec(Vect0r.fromList(combined)));
                }
            } else {
                java.util.List<Cell> combined = new java.util.ArrayList<>(lrow.columnCount()+right.columnCount());
                for(int i=0;i<lrow.columnCount();i++) combined.add(lrow.get(i));
                // add nulls for right side
                for(int j=0;j<right.columnCount();j++) {
                    final int idx = j;
                    combined.add(new Cell(null, new com.moneyfan.core.CellMeta(() -> rightScalars.get(idx))));
                }
                outRows.add(new RowVec(Vect0r.fromList(combined)));
            }
        }
        return new GridCursor(Vect0r.fromList(outRows));
    }
}