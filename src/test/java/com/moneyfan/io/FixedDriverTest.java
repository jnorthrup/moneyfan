package com.moneyfan.io;

import com.moneyfan.core.IOMemento;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.time.Instant;
import java.time.LocalDate;

import static org.junit.jupiter.api.Assertions.*;

public class FixedDriverTest {

    @Test
    void intDriverReadWrite() {
        @SuppressWarnings("unchecked")
        FixedDriver<Integer> driver = (FixedDriver<Integer>) FixedDriver.MAPPED_DRIVERS.get(IOMemento.IO_INT);
        ByteBuffer buf = ByteBuffer.allocate(driver.size());
        driver.write(buf, 0, 42);
        int val = driver.read(buf, 0);
        assertEquals(42, val);
    }

    @Test
    void localDateDriverReadWrite() {
        @SuppressWarnings("unchecked")
        FixedDriver<LocalDate> driver = (FixedDriver<LocalDate>) FixedDriver.MAPPED_DRIVERS.get(IOMemento.IO_LOCAL_DATE);
        ByteBuffer buf = ByteBuffer.allocate(driver.size());
        LocalDate date = LocalDate.of(2023, 1, 1);
        driver.write(buf, 0, date);
        LocalDate read = driver.read(buf, 0);
        assertEquals(date, read);
    }

    @Test
    void instantDriverReadWrite() {
        @SuppressWarnings("unchecked")
        FixedDriver<Instant> driver = (FixedDriver<Instant>) FixedDriver.MAPPED_DRIVERS.get(IOMemento.IO_INSTANT);
        ByteBuffer buf = ByteBuffer.allocate(driver.size());
        Instant now = Instant.now().truncatedTo(java.time.temporal.ChronoUnit.MILLIS);
        driver.write(buf, 0, now);
        Instant read = driver.read(buf, 0);
        assertEquals(now.getEpochSecond(), read.getEpochSecond());
        assertEquals(now.getNano(), read.getNano());
    }
}