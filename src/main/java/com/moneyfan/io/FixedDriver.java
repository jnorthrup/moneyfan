package com.moneyfan.io;

import com.moneyfan.core.IOMemento;

import java.nio.ByteBuffer;
import java.time.Instant;
import java.time.LocalDate;
import java.util.Map;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;

/**
 * CellDriver implementation for fixed-size primitive types.
 * Uses functional read/write lambdas to parameterize behavior.
 */
public record FixedDriver<T>(int size,
                             BiFunction<ByteBuffer, Integer, T> reader,
                             TriConsumer<ByteBuffer, Integer, T> writer) implements CellDriver<T> {

    @FunctionalInterface
    public interface TriConsumer<A, B, C> {
        void accept(A a, B b, C c);
    }

    @Override
    public T read(ByteBuffer buffer, int offset) {
        return reader.apply(buffer, offset);
    }

    @Override
    public void write(ByteBuffer buffer, int offset, T value) {
        writer.accept(buffer, offset, value);
    }

    private static final FixedDriver<Integer> INT_DRIVER = new FixedDriver<>(4,
            (buf, off) -> buf.getInt(off),
            (buf, off, v) -> buf.putInt(off, v));

    private static final FixedDriver<Long> LONG_DRIVER = new FixedDriver<>(8,
            (buf, off) -> buf.getLong(off),
            (buf, off, v) -> buf.putLong(off, v));

    private static final FixedDriver<Double> DOUBLE_DRIVER = new FixedDriver<>(8,
            (buf, off) -> buf.getDouble(off),
            (buf, off, v) -> buf.putDouble(off, v));

    private static final FixedDriver<LocalDate> LOCALDATE_DRIVER = new FixedDriver<>(4,
            (buf, off) -> LocalDate.ofEpochDay(buf.getInt(off)),
            (buf, off, v) -> buf.putInt(off, (int) v.toEpochDay()));

    private static final FixedDriver<Instant> INSTANT_DRIVER = new FixedDriver<>(12,
            (buf, off) -> {
                long sec = buf.getLong(off);
                int nanos = buf.getInt(off + 8);
                return Instant.ofEpochSecond(sec, nanos);
            },
            (buf, off, v) -> {
                buf.putLong(off, v.getEpochSecond());
                buf.putInt(off + 8, v.getNano());
            });

    public static final Map<IOMemento, FixedDriver<?>> MAPPED_DRIVERS = Map.of(
            IOMemento.IO_INT, INT_DRIVER,
            IOMemento.IO_LONG, LONG_DRIVER,
            IOMemento.IO_DOUBLE, DOUBLE_DRIVER,
            IOMemento.IO_LOCAL_DATE, LOCALDATE_DRIVER,
            IOMemento.IO_INSTANT, INSTANT_DRIVER
    );
}