package moneyfan.grid;

import java.util.List;
import java.util.stream.Collectors;
import moneyfan.core.Scalar;

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
        return rows.get(0).cells().accessor()
            .andThen(cell -> cell.meta().provider().get())
            .apply(0).stream().collect(Collectors.toList());
    }
}