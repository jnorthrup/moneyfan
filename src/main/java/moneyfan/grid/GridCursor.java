package moneyfan.grid;

import java.util.List;
import moneyfan.core.Scalar;
import moneyfan.grid.Cell;
import moneyfan.grid.RowVec;
import moneyfan.grid.Vect0r;

public record GridCursor(Vect0r<RowVec> rows) {
    public int rowCount() {
        return rows.size();
    }

    public int columnCount() {
        return rows.size() > 0 ? rows.get(0).cells().size() : 0;
    }

    public RowVec getRow(int index) {
        return rows.get(index);
    }

    public List<Scalar> getScalars() {
        if (rowCount() == 0) return List.of();
        RowVec firstRow = rows.get(0);
        Vect0r<Cell> cells = firstRow.cells();
        java.util.ArrayList<Scalar> scalars = new java.util.ArrayList<>(cells.size());
        for (int i = 0; i < cells.size(); i++) {
            scalars.add(cells.get(i).meta().provider().get());
        }
        return scalars;
    }
}