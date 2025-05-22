package moneyfan.grid;

import moneyfan.core.*;
import org.junit.jupiter.api.Test;
import java.util.List;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.*;

class GridCursorTest {
    @Test
    void testGetScalars() {
        Scalar s1 = new Scalar(IOMemento.IO_INT, "id");
        Scalar s2 = new Scalar(IOMemento.IO_STRING_FIXED, "symbol");
        CellMeta m1 = new CellMeta(() -> s1);
        CellMeta m2 = new CellMeta(() -> s2);
        Cell c1 = new Cell(1, m1);
        Cell c2 = new Cell("BTCUSDT", m2);
        RowVec row = new RowVec(Vect0r.fromList(List.of(c1, c2)));
        GridCursor cursor = new GridCursor(Vect0r.fromList(List.of(row)));
        List<Scalar> scalars = cursor.getScalars();
        assertEquals(2, scalars.size());
        assertEquals(s1, scalars.get(0));
        assertEquals(s2, scalars.get(1));
    }
}