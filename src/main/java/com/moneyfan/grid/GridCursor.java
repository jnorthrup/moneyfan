package com.moneyfan.grid;

import java.util.Objects;

/**
 * A 2D grid abstraction represented as a vector of rows.
 */
public record GridCursor(Vect0r<RowVec> rows) {

    public GridCursor {
        Objects.requireNonNull(rows, "rows");
    }

    public int rowCount() {
        return rows.size();
    }

    public int columnCount() {
        return rowCount() == 0 ? 0 : rows.get(0).cells().size();
    }

    public RowVec getRow(int index) {
        return rows.get(index);
    }

    /**
     * Returns a vector of {@link com.moneyfan.core.Scalar} describing the grid schema.
     */
    public com.moneyfan.grid.Vect0r<com.moneyfan.core.Scalar> getScalars() {
        if (rowCount() == 0) {
            return Vect0r.of(0, i -> null);
        }
        RowVec firstRow = rows.get(0);
        Vect0r<Cell> firstCells = firstRow.cells();
        return Vect0r.of(firstCells.size(), i -> firstCells.get(i).meta().provider().get());
    }
}