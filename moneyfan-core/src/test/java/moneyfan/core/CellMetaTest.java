package moneyfan.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.util.function.Supplier;

class CellMetaTest {
    @Test
    void testProvider() {
        Scalar scalar = new Scalar(IOMemento.IO_LONG, "baz");
        Supplier<Scalar> supplier = () -> scalar;
        CellMeta meta = new CellMeta(supplier);
        assertSame(scalar, meta.provider().get());
    }
}