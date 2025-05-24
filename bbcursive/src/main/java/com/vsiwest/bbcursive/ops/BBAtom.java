package com.vsiwest.bbcursive.ops;

import com.vsiwest.bbcursive.core.Cursive;
import com.vsiwest.bbcursive.core.ParseResult;
import org.jetbrains.annotations.NotNull;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

/**
 * {@code BBAtom} provides fundamental, atomic parsing operations directly on {@link ByteBuffer}s.
 * These operations are typically low-level and serve as building blocks for more complex parsers
 * via {@link BBCombinator}. Each method in this enum represents a basic parsing action.
 *
 * This enum acts as a utility class, encapsulating related static-like methods.
 * Note: Enum entries are typically singletons; here, we leverage the enum's static-like nature
 * to group utility methods rather than creating actual instances.
 */
public enum BBAtom {
    // This enum doesn't need instances for utility methods.
    // It's used as a namespace for static-like methods, aligning with the "Encapsulation via Enums" principle.
    ; // No instances

    /**
     * Parses a single byte from the buffer.
     * @return A Cursive parser for a single Byte.
     */
    public static @NotNull Cursive<Byte> readByte() {
        return buffer -> {
            if (buffer.hasRemaining()) {
                byte b = buffer.get();
                return ParseResult.success(b, buffer);
            }
            return ParseResult.failure();
        };
    }

    /**
     * Parses a single integer (4 bytes) from the buffer.
     * @return A Cursive parser for an Integer.
     */
    public static @NotNull Cursive<Integer> readInt() {
        return buffer -> {
            if (buffer.remaining() >= Integer.BYTES) {
                int i = buffer.getInt();
                return ParseResult.success(i, buffer);
            }
            return ParseResult.failure();
        };
    }

    /**
     * Parses a single long (8 bytes) from the buffer.
     * @return A Cursive parser for a Long.
     */
    public static @NotNull Cursive<Long> readLong() {
        return buffer -> {
            if (buffer.remaining() >= Long.BYTES) {
                long l = buffer.getLong();
                return ParseResult.success(l, buffer);
            }
            return ParseResult.failure();
        };
    }

    /**
     * Reads a fixed number of bytes from the buffer and returns them as a new ByteBuffer slice.
     * The returned buffer is a slice of the original buffer, meaning it shares the same backing array.
     * Its position will be 0 and its limit will be {@code length}. The original buffer's position
     * will be advanced by {@code length}.
     *
     * @param length The number of bytes to read.
     * @return A Cursive parser for a ByteBuffer slice.
     */
    public static @NotNull Cursive<ByteBuffer> readSlice(int length) {
        return buffer -> {
            if (buffer.remaining() >= length) {
                int originalLimit = buffer.limit();
                buffer.limit(buffer.position() + length);
                ByteBuffer slice = buffer.slice(); // Creates a new buffer, sharing content. Position is 0, limit is length.
                buffer.position(buffer.limit()); // Advance original buffer's position
                buffer.limit(originalLimit); // Restore original limit
                return ParseResult.success(slice, buffer);
            }
            return ParseResult.failure();
        };
    }

    /**
     * Reads a fixed number of bytes from the buffer and decodes them as a UTF-8 String.
     * @param length The number of bytes to read for the string.
     * @return A Cursive parser for a String.
     */
    public static @NotNull Cursive<String> readString(int length) {
        return readSlice(length).map(slice -> {
            byte[] bytes = new byte[slice.remaining()];
            slice.get(bytes); // Copies bytes from slice to array
            return new String(bytes, StandardCharsets.UTF_8);
        });
    }

    // Example of a 'match' parser
    /**
     * Parses a specific sequence of bytes.
     * @param expectedBytes The byte sequence to match.
     * @return A Cursive parser that succeeds if the sequence matches.
     */
    public static @NotNull Cursive<ByteBuffer> matchBytes(@NotNull byte[] expectedBytes) {
        return buffer -> {
            if (buffer.remaining() >= expectedBytes.length) {
                int originalPosition = buffer.position();
                for (byte expectedByte : expectedBytes) {
                    if (buffer.get() != expectedByte) {
                        buffer.position(originalPosition); // Rewind on mismatch
                        return ParseResult.failure();
                    }
                }
                return ParseResult.success(ByteBuffer.wrap(expectedBytes), buffer); // Return a copy of matched bytes
            }
            return ParseResult.failure();
        };
    }
}
