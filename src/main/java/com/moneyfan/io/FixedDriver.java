package com.moneyfan.io;

import com.moneyfan.core.IOMemento;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.LocalDate;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * CellDriver for fixed-size primitive types.
 */
public record FixedDriver<Value>(
    BiFunction<ByteBuffer, Integer, Value> reader,
    TriConsumer<ByteBuffer, Integer, Value> writer,
    int size
) implements CellDriver<ByteBuffer, Value> {
    
    @Override
    public Value read(ByteBuffer buffer, int offset) {
        return reader.apply(buffer, offset);
    }
    
    @Override
    public void write(ByteBuffer buffer, int offset, Value value) {
        writer.accept(buffer, offset, value);
    }
    
    @FunctionalInterface
    public interface TriConsumer<T, U, V> {
        void accept(T t, U u, V v);
    }
    
    // Static mapping from IOMemento to appropriate FixedDriver
    @SuppressWarnings("unchecked")
    public static final Map<IOMemento, FixedDriver<?>> MAPPED_DRIVERS = Map.of(
        IOMemento.IO_INT, new FixedDriver<>(
            ByteBuffer::getInt,
            ByteBuffer::putInt,
            4
        ),
        
        IOMemento.IO_LONG, new FixedDriver<>(
            ByteBuffer::getLong,
            ByteBuffer::putLong,
            8
        ),
        
        IOMemento.IO_DOUBLE, new FixedDriver<>(
            ByteBuffer::getDouble,
            ByteBuffer::putDouble,
            8
        ),
        
        IOMemento.IO_LOCAL_DATE, new FixedDriver<>(
            (buffer, offset) -> LocalDate.ofEpochDay(buffer.getLong(offset)),
            (buffer, offset, date) -> buffer.putLong(offset, ((LocalDate) date).toEpochDay()),
            8
        ),
        
        IOMemento.IO_INSTANT, new FixedDriver<>(
            (buffer, offset) -> Instant.ofEpochMilli(buffer.getLong(offset)),
            (buffer, offset, instant) -> buffer.putLong(offset, ((Instant) instant).toEpochMilli()),
            8
        )
    );
    
    // Factory method for string drivers of specific length
    public static FixedDriver<String> stringDriver(int length) {
        return new FixedDriver<>(
            (buffer, offset) -> {
                byte[] bytes = new byte[length];
                ByteBuffer slice = buffer.duplicate();
                slice.position(offset);
                slice.get(bytes, 0, length);
                return new String(bytes, StandardCharsets.UTF_8).trim();
            },
            (buffer, offset, value) -> {
                byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
                ByteBuffer slice = buffer.duplicate();
                slice.position(offset);
                slice.put(bytes, 0, Math.min(bytes.length, length));
                // Pad with spaces if needed
                for (int i = bytes.length; i < length; i++) {
                    slice.put((byte) 32); // space character
                }
            },
            length
        );
    }
    
    @SuppressWarnings("unchecked")
    public static <T> FixedDriver<T> getDriver(IOMemento type, int stringLength) {
        if (type == IOMemento.IO_STRING_FIXED) {
            return (FixedDriver<T>) stringDriver(stringLength);
        }
        return (FixedDriver<T>) MAPPED_DRIVERS.get(type);
    }
}