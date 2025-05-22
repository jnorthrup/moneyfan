package com.moneyfan.core;

import java.nio.ByteBuffer;
import java.time.Instant;
import java.time.LocalDate;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * Enum for data types and their binary characteristics.
 */
public enum IOMemento {
    IO_INT("int", 4, 
        (bb, offset) -> bb.getInt(offset), 
        (bb, offset, value) -> { bb.putInt(offset, (Integer) value); return null; },
        Object::toString,
        Integer::valueOf),
    
    IO_LONG("long", 8, 
        (bb, offset) -> bb.getLong(offset), 
        (bb, offset, value) -> { bb.putLong(offset, (Long) value); return null; },
        Object::toString,
        Long::valueOf),
    
    IO_DOUBLE("double", 8, 
        (bb, offset) -> bb.getDouble(offset), 
        (bb, offset, value) -> { bb.putDouble(offset, (Double) value); return null; },
        Object::toString,
        Double::valueOf),
    
    IO_LOCAL_DATE("localDate", 8, 
        (bb, offset) -> LocalDate.ofEpochDay(bb.getLong(offset)), 
        (bb, offset, value) -> { bb.putLong(offset, ((LocalDate) value).toEpochDay()); return null; },
        Object::toString,
        LocalDate::parse),
    
    IO_INSTANT("instant", 8, 
        (bb, offset) -> Instant.ofEpochMilli(bb.getLong(offset)), 
        (bb, offset, value) -> { bb.putLong(offset, ((Instant) value).toEpochMilli()); return null; },
        Object::toString,
        Instant::parse),
    
    IO_STRING_FIXED("string", -1, 
        (bb, offset) -> {
            // Determined at runtime based on metadata
            throw new UnsupportedOperationException("Direct read not supported for fixed string");
        }, 
        (bb, offset, value) -> {
            // Determined at runtime based on metadata
            throw new UnsupportedOperationException("Direct write not supported for fixed string");
        },
        Object::toString,
        s -> s);
    
    private final String typeName;
    private final int size;
    private final BiFunction<ByteBuffer, Integer, Object> reader;
    private final TriFunction<ByteBuffer, Integer, Object, Void> writer;
    private final Function<Object, String> toString;
    private final Function<String, Object> fromString;
    
    IOMemento(String typeName, int size, 
              BiFunction<ByteBuffer, Integer, Object> reader, 
              TriFunction<ByteBuffer, Integer, Object, Void> writer,
              Function<Object, String> toString,
              Function<String, Object> fromString) {
        this.typeName = typeName;
        this.size = size;
        this.reader = reader;
        this.writer = writer;
        this.toString = toString;
        this.fromString = fromString;
    }
    
    public String getTypeName() {
        return typeName;
    }
    
    public int getSize() {
        return size;
    }
    
    public Object read(ByteBuffer buffer, int offset) {
        return reader.apply(buffer, offset);
    }
    
    public void write(ByteBuffer buffer, int offset, Object value) {
        writer.apply(buffer, offset, value);
    }
    
    public String convertToString(Object value) {
        return toString.apply(value);
    }
    
    public Object convertFromString(String value) {
        return fromString.apply(value);
    }
    
    @FunctionalInterface
    private interface TriFunction<T, U, V, R> {
        R apply(T t, U u, V v);
    }
}