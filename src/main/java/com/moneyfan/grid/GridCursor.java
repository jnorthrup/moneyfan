package com.moneyfan.grid;

import com.moneyfan.core.Scalar;

import java.util.List;
import java.util.Objects;
import java.util.stream.IntStream;

/**
 * Two-dimensional grid abstraction backed by a lazy vector of rows.
 */
public record GridCursor(Vect0r<RowVec> rows) {

    public GridCursor {
        Objects.requireNonNull(rows, "rows");
        // sanity: ensure all rows have the same column count
        if (rows.size() > 1) {
            int cols = rows.get(0).size();
            for (int i = 1; i < rows.size(); i++) {
                if (rows.get(i).size() != cols) {
                    throw new IllegalArgumentException("Inconsistent column count at row " + i);
                }
            }
        }
    }

    public int rowCount() {return rows.size();}

    public int columnCount() {return rows.size() == 0 ? 0 : rows.get(0).size();}

    public RowVec row(int index) {return rows.get(index);} // Index checks delegated

    public List<Scalar> scalars() {
        return IntStream.range(0, columnCount())
                .mapToObj(c -> rows.get(0).scalar(c))
                .toList();
    }
}