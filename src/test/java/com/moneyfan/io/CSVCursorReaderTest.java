package com.moneyfan.io;

import com.moneyfan.core.IOMemento;
import com.moneyfan.core.Scalar;
import com.moneyfan.grid.GridCursor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class CSVCursorReaderTest {

    @Test
    void readsCsvIntoGrid(@TempDir Path tempDir) throws Exception {
        Path csv = tempDir.resolve("sample.csv");
        Files.writeString(csv, "1,alice,10.5\n2,bob,20.0\n");
        List<Scalar> schema = List.of(
                Scalar.of(IOMemento.IO_INT, "id"),
                Scalar.of(IOMemento.IO_STRING_FIXED, "name"),
                Scalar.of(IOMemento.IO_DOUBLE, "value")
        );
        GridCursor grid = CSVCursorReader.read(csv, schema);
        assertEquals(2, grid.rowCount());
        assertEquals(3, grid.columnCount());
        assertEquals("bob", grid.getRow(1).get(1).value());
    }
}