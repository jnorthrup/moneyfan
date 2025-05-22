package com.moneyfan.io;

import com.moneyfan.core.IOMemento;
import com.moneyfan.core.Scalar;
import com.moneyfan.grid.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class ISAMReadWriteTest {

    @Test
    void writeThenRead(@TempDir Path tempDir) throws Exception {
        // Build simple grid
        List<Scalar> schema = List.of(
                Scalar.of(IOMemento.IO_INT, "id"),
                Scalar.of(IOMemento.IO_STRING_FIXED, "name"),
                Scalar.of(IOMemento.IO_DOUBLE, "value")
        );
        RowVec row1 = makeRow(1, "alice", 10.5, schema);
        RowVec row2 = makeRow(2, "bob", 20.0, schema);
        GridCursor grid = new GridCursor(Vect0r.fromList(List.of(row1, row2)));

        Path dataFile = tempDir.resolve("data.bin");
        ISAMWriter.write(grid, dataFile);

        try(ISAMReader reader = new ISAMReader(dataFile)) {
            GridCursor read = reader.open();
            assertEquals(2, read.rowCount());
            assertEquals("bob", read.getRow(1).get(1).value());
            assertEquals(10.5, (Double) read.getRow(0).get(2).value(), 1e-6);
        }
    }

    private static RowVec makeRow(int id, String name, double value, List<Scalar> schema) {
        List<Cell> cells = List.of(
                new Cell(id, new com.moneyfan.core.CellMeta(() -> schema.get(0))),
                new Cell(name, new com.moneyfan.core.CellMeta(() -> schema.get(1))),
                new Cell(value, new com.moneyfan.core.CellMeta(() -> schema.get(2)))
        );
        return new RowVec(Vect0r.fromList(cells));
    }
}