package borg.trikeshed.nio; // Moved package

import com.yourdomain.bbcursive.ops.BBAtom;
import com.yourdomain.bbcursive.core.Cursive;
import com.yourdomain.bbcursive.ops.BBCombinator;
import borg.trikeshed.lib.Series; // Added for IoInstant decoder
// Removed: import com.yourdomain.bikeshed.type.TypeMemento;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.nio.ByteBuffer;
import java.time.Instant;
import java.time.LocalDate;
// Removed: import java.time.ZoneOffset; // Not directly used in the copied logic
import java.util.function.Function;

/**
 * Defines data types and their corresponding serialization/deserialization strategies
 * using {@code bbcursive} for efficient {@link ByteBuffer} operations.
 * This enum integrates directly with ISAM for fixed-format file I/O.
 */
public enum IOMemento { // Removed "implements TypeMemento"
    IoByte(Byte.BYTES),
    IoShort(Short.BYTES),
    IoInt(Integer.BYTES),
    IoLong(Long.BYTES),
    IoFloat(Float.BYTES),
    IoDouble(Double.BYTES),
    IoBoolean(Byte.BYTES), // Typically 1 byte
    IoChar(Character.BYTES), // Typically 2 bytes for UTF-16 char
    IoInstant(Long.BYTES + Integer.BYTES), // Epoch seconds (long) + nanoseconds (int) = 12 bytes
    IoLocalDate(Long.BYTES), // Epoch days (long) for simplicity
    IoString(null), // Variable length, requires explicit length config
    IoByteArray(null); // Variable length

    private final Integer networkSize;

    IOMemento(@Nullable Integer networkSize) {
        this.networkSize = networkSize;
    }

    // @Override // No longer overriding from TypeMemento
    public Integer networkSize() {
        return networkSize;
    }

    /**
     * Provides a decoder function for this type, consuming bytes from a ByteBuffer
     * and returning the corresponding Java object.
     * Uses {@code bbcursive} for low-level parsing.
     *
     * @return A Function that takes a ByteBuffer and returns the decoded object.
     * @throws IllegalStateException if {@code networkSize} is null for a fixed-size type.
     */
    public @NotNull Cursive<Object> createDecoder() {
        if (networkSize == null) {
            // For variable length types, createDecoder(length) must be called.
            // Or, if we want a default behavior for truly variable (e.g. read until end), it's not defined here.
            throw new IllegalStateException("Variable-length types require a specific length to create a decoder. Call createDecoder(length).");
        }
        return createDecoder(networkSize);
    }

    /**
     * Provides a decoder function for this type, consuming {@code length} bytes from a ByteBuffer
     * and returning the corresponding Java object.
     *
     * @param length The number of bytes to decode. Required for variable-length types.
     * @return A Function that takes a ByteBuffer and returns the decoded object.
     */
    public @NotNull Cursive<Object> createDecoder(int length) {
        return switch (this) {
            case IoByte -> BBAtom.readByte().map(b -> (Object) b);
            case IoShort -> BBAtom.readSlice(Short.BYTES).map(ByteBuffer::getShort).map(s -> (Object) s);
            case IoInt -> BBAtom.readInt().map(i -> (Object) i);
            case IoLong -> BBAtom.readLong().map(l -> (Object) l);
            case IoFloat -> BBAtom.readSlice(Float.BYTES).map(ByteBuffer::getFloat).map(f -> (Object) f);
            case IoDouble -> BBAtom.readSlice(Double.BYTES).map(ByteBuffer::getDouble).map(d -> (Object) d);
            case IoBoolean -> BBAtom.readByte().map(b -> (Object) (b != 0));
            case IoChar -> BBAtom.readSlice(Character.BYTES).map(ByteBuffer::getChar).map(c -> (Object) c);
            case IoInstant -> BBCombinator.sequence(BBAtom.readLong(), BBAtom.readInt())
                    .map(series -> (Object) Instant.ofEpochSecond((Long) series.get(0), (Integer) series.get(1))); // Changed list to series
            case IoLocalDate -> BBAtom.readLong().map(epochDay -> (Object) LocalDate.ofEpochDay((Long) epochDay));
            case IoString -> BBAtom.readString(length).map(s -> (Object) s);
            case IoByteArray -> BBAtom.readSlice(length).map(buffer -> {
                byte[] bytes = new byte[buffer.remaining()];
                buffer.get(bytes);
                return (Object) bytes;
            });
        };
    }

    /**
     * Provides an encoder function for this type, converting a Java object
     * into a ByteBuffer that can be written to storage.
     *
     * @return A Function that takes an object and returns a ByteBuffer containing its encoded form.
     */
    public @NotNull Function<Object, ByteBuffer> createEncoder() {
        if (networkSize == null) {
            // Similar to createDecoder, variable length types need explicit length for this encoder.
            throw new IllegalStateException("Variable-length types require a specific length to create an encoder. Call createEncoder(length).");
        }
        return createEncoder(networkSize);
    }

    /**
     * Provides an encoder function for this type, converting a Java object
     * into a ByteBuffer of {@code length} bytes that can be written to storage.
     * This is useful for fixed-size fields where variable-length data needs padding/truncation.
     *
     * @param length The fixed size (in bytes) of the output buffer.
     * @return A Function that takes an object and returns a ByteBuffer containing its encoded form.
     */
    public @NotNull Function<Object, ByteBuffer> createEncoder(int length) {
        return value -> {
            // Ensure buffer is allocated with 'length' for fixed-size semantics
            ByteBuffer buffer = ByteBuffer.allocate(length);
            switch (this) {
                case IoByte:
                    buffer.put((Byte) value);
                    break;
                case IoShort:
                    buffer.putShort((Short) value);
                    break;
                case IoInt:
                    buffer.putInt((Integer) value);
                    break;
                case IoLong:
                    buffer.putLong((Long) value);
                    break;
                case IoFloat:
                    buffer.putFloat((Float) value);
                    break;
                case IoDouble:
                    buffer.putDouble((Double) value);
                    break;
                case IoBoolean:
                    buffer.put((byte) (((Boolean) value) ? 1 : 0));
                    break;
                case IoChar:
                    buffer.putChar((Character) value);
                    break;
                case IoInstant:
                    Instant instant = (Instant) value;
                    // Ensure we do not overflow the allocated buffer if length is too small for Instant
                    if (length >= (Long.BYTES + Integer.BYTES)) {
                        buffer.putLong(instant.getEpochSecond());
                        buffer.putInt(instant.getNano());
                    } else {
                        // Handle error or partial write for Instant if length is not sufficient
                        // For simplicity, this example might truncate or throw.
                        // Or, rely on ByteBuffer's BufferOverflowException if not careful.
                    }
                    break;
                case IoLocalDate:
                    LocalDate date = (LocalDate) value;
                     if (length >= Long.BYTES) { // Ensure space for epoch day
                        buffer.putLong(date.toEpochDay());
                    }
                    break;
                case IoString:
                    byte[] strBytes = ((String) value).getBytes(java.nio.charset.StandardCharsets.UTF_8);
                    // Pad or truncate string bytes to fit 'length'
                    int bytesToCopyString = Math.min(strBytes.length, length);
                    buffer.put(strBytes, 0, bytesToCopyString);
                    // If strBytes.length < length, the rest of the buffer is zero-filled by allocate.
                    break;
                case IoByteArray:
                    byte[] rawBytes = (byte[]) value;
                    // Pad or truncate rawBytes to fit 'length'
                    int bytesToCopyRaw = Math.min(rawBytes.length, length);
                    buffer.put(rawBytes, 0, bytesToCopyRaw);
                    // If rawBytes.length < length, the rest of the buffer is zero-filled.
                    break;
            }
            buffer.flip(); // Prepare for reading from the buffer
            return buffer;
        };
    }
}
