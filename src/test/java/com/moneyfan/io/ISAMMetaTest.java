package com.moneyfan.io;

import com.moneyfan.core.IOMemento;
import com.moneyfan.core.Scalar;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class ISAMMetaTest {

    @Test
    void writeAndReadMeta(@TempDir Path tempDir) throws Exception {
        List<Scalar> cols = List.of(
                Scalar.of(IOMemento.IO_INT, "id"),
                Scalar.of(IOMemento.IO_STRING_FIXED, "name"),
                Scalar.of(IOMemento.IO_DOUBLE, "value")
        );
        List<Integer> fixedLens = List.of(-1, 20, -1);
        ISAMMeta meta = ISAMMeta.fromColumns(cols, fixedLens);
        Path file = tempDir.resolve("sample.meta");
        meta.write(file);
        ISAMMeta read = ISAMMeta.read(file);
        assertEquals(meta.recordLength(), read.recordLength());
        assertEquals(3, read.columnCount());
        assertEquals(20, read.fixedStringLength(1));
        assertEquals(meta.offset(2), read.offset(2));
    }
}