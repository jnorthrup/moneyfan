package com.example.bikeshed.types;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.LocalDate;
import java.util.function.Function;

/**
 * Defines standard data types and their corresponding serialization/deserialization strategies
 * for ISAM integration. Each enum entry specifies `networkSize` where applicable.
 */
public enum IOMemento implements TypeMemento {
    IO_BOOLEAN(1, byteBuffer -> byteBuffer.get() != 0, (value, byteBuffer) -> byteBuffer.put((byte) (((Boolean) value) ? 1 : 0))),
    IO_BYTE(1, ByteBuffer::get, (value, byteBuffer) -> byteBuffer.put((Byte) value)),
    IO_SHORT(2, ByteBuffer::getShort, (value, byteBuffer) -> byteBuffer.putShort((Short) value)),
    IO_INT(4, ByteBuffer::getInt, (value, byteBuffer) -> byteBuffer.putInt((Integer) value)),
    IO_LONG(8, ByteBuffer::getLong, (value, byteBuffer) -> byteBuffer.putLong((Long) value)),
    IO_FLOAT(4, ByteBuffer::getFloat, (value, byteBuffer) -> byteBuffer.putFloat((Float) value)),
    IO_DOUBLE(8, ByteBuffer::getDouble, (value, byteBuffer) -> byteBuffer.putDouble((Double) value)),

    /**
     * Represents a `LocalDate`. Serialized as 8 bytes (epoch days as a long).
     */
    IO_LOCAL_DATE(8, byteBuffer -> LocalDate.ofEpochDay(byteBuffer.getLong()),
            (value, byteBuffer) -> byteBuffer.putLong(((LocalDate) value).toEpochDay())),

    /**
     * Represents an `Instant`. Serialized as 12 bytes (8 bytes for epoch seconds, 4 bytes for nano adjustment).
     */
    IO_INSTANT(12, byteBuffer -> {
        long epochSeconds = byteBuffer.getLong();
        int nanoAdjustment = byteBuffer.getInt();
        return Instant.ofEpochSecond(epochSeconds, nanoAdjustment);
    }, (value, byteBuffer) -> {
        Instant instant = (Instant) value;
        byteBuffer.putLong(instant.getEpochSecond());
        byteBuffer.putInt(instant.getNano());
    }),

    /**
     * Represents a fixed-length string. `networkSize` is `null` here, implying it must be
     * configured externally (e.g., via `Map<String, Int>` in `IsamMetaFileReader.write`).
     * Stored as UTF-8 bytes.
     */
    IO_STRING(null, byteBuffer -> {
        // This reader needs to know the length. It expects the buffer to be sized for the string.
        // For ISAM, string length would be pre-determined per column definition.
        byte[] bytes = new byte[byteBuffer.remaining()];
        byteBuffer.get(bytes);
        return new String(bytes, StandardCharsets.UTF_8);
    }, (value, byteBuffer) -> {
        byte[] bytes = ((String) value).getBytes(StandardCharsets.UTF_8);
        byteBuffer.put(bytes);
        // Pad with zeros if necessary to fill fixed size
        if (byteBuffer.hasRemaining()) {
            byteBuffer.put(new byte[byteBuffer.remaining()]);
        }
    });

    private final Integer networkSize;
    private final Function<ByteBuffer, Object> decoder;
    private final BiConsumer<Object, ByteBuffer> encoder;

    IOMemento(Integer networkSize, Function<ByteBuffer, Object> decoder, BiConsumer<Object, ByteBuffer> encoder) {
        this.networkSize = networkSize;
        this.decoder = decoder;
        this.encoder = encoder;
    }

    @Override
    public Integer getNetworkSize() {
        return networkSize;
    }

    /**
     * Returns a decoder function for this type.
     * @return A function that takes a ByteBuffer and decodes an object of this type.
     */
    public Function<ByteBuffer, Object> getDecoder() {
        return decoder;
    }

    /**
     * Returns an encoder function for this type.
     * @return A function that takes an object and a ByteBuffer and encodes the object into the buffer.
     */
    public BiConsumer<Object, ByteBuffer> getEncoder() {
        return encoder;
    }

    // Custom functional interface for encoding
    @FunctionalInterface
    public interface BiConsumer<T, U> {
        void accept(T t, U u);
    }

    /**
     * Helper to parse string representation of type to its corresponding IOMemento.
     * @param typeName String name of the type.
     * @return IOMemento enum value.
     * @throws IllegalArgumentException if typeName does not correspond to an IOMemento.
     */
    public static IOMemento fromTypeName(String typeName) {
        return IOMemento.valueOf(typeName.toUpperCase().replace(" ", "_"));
    }
}
