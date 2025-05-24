package com.example.bbcursive;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Objects;
import java.util.Optional;

/**
 * Provides atomic parsing operations for ByteBuffer, acting as fundamental building blocks.
 * These are simple, direct byte-level parsers.
 * Each method returns a `Cursive` parser.
 */
public enum BBAtom implements Cursive<Object> {
    // Enum members for different atomic parsing operations
    READ_BYTE {
        @Override
        public Byte apply(ByteBuffer buffer) {
            if (buffer.hasRemaining()) {
                return buffer.get();
            }
            return null; // Or throw NoSuchElementException
        }
    },
    READ_SHORT {
        @Override
        public Short apply(ByteBuffer buffer) {
            if (buffer.remaining() >= 2) {
                return buffer.getShort();
            }
            return null;
        }
    },
    READ_INT {
        @Override
        public Integer apply(ByteBuffer buffer) {
            if (buffer.remaining() >= 4) {
                return buffer.getInt();
            }
            return null;
        }
    },
    READ_LONG {
        @Override
        public Long apply(ByteBuffer buffer) {
            if (buffer.remaining() >= 8) {
                return buffer.getLong();
            }
            return null;
        }
    },
    READ_FLOAT {
        @Override
        public Float apply(ByteBuffer buffer) {
            if (buffer.remaining() >= 4) {
                return buffer.getFloat();
            }
            return null;
        }
    },
    READ_DOUBLE {
        @Override
        public Double apply(ByteBuffer buffer) {
            if (buffer.remaining() >= 8) {
                return buffer.getDouble();
            }
            return null;
        }
    };

    /**
     * Default implementation for enum methods. Concrete implementations are in enum constants.
     * @param buffer The ByteBuffer to parse.
     * @return The parsed object.
     */
    @Override
    public abstract Object apply(ByteBuffer buffer);

    // Factory methods to create specific Cursive instances from BBAtom operations
    public static Cursive<Byte> byteP() {
        return (Cursive<Byte>) READ_BYTE;
    }

    public static Cursive<Short> shortP() {
        return (Cursive<Short>) READ_SHORT;
    }

    public static Cursive<Integer> intP() {
        return (Cursive<Integer>) READ_INT;
    }

    public static Cursive<Long> longP() {
        return (Cursive<Long>) READ_LONG;
    }

    public static Cursive<Float> floatP() {
        return (Cursive<Float>) READ_FLOAT;
    }

    public static Cursive<Double> doubleP() {
        return (Cursive<Double>) READ_DOUBLE;
    }

    /**
     * Creates a parser that matches a specific byte value.
     *
     * @param expectedByte The byte value to match.
     * @return A Cursive parser that returns the matched byte if successful, null otherwise.
     */
    public static Cursive<Byte> matchByte(byte expectedByte) {
        return buffer -> {
            if (buffer.hasRemaining() && buffer.get(buffer.position()) == expectedByte) {
                return buffer.get(); // Consume the byte
            }
            return null;
        };
    }

    /**
     * Creates a parser that matches a specific sequence of bytes (a literal).
     *
     * @param literal The byte array literal to match.
     * @return A Cursive parser that returns the literal as a byte array if successful, null otherwise.
     */
    public static Cursive<byte[]> matchLiteral(byte[] literal) {
        return buffer -> {
            if (buffer.remaining() < literal.length) {
                return null;
            }
            int originalPos = buffer.position();
            byte[] matched = new byte[literal.length];
            buffer.get(matched);
            for (int i = 0; i < literal.length; i++) {
                if (matched[i] != literal[i]) {
                    buffer.position(originalPos); // Rewind on mismatch
                    return null;
                }
            }
            return matched;
        };
    }

    /**
     * Creates a parser that reads a fixed number of bytes and returns them as a ByteBuffer slice.
     * The returned ByteBuffer will share content with the original but have its own position/limit.
     *
     * @param length The number of bytes to read.
     * @return A Cursive parser that returns a ByteBuffer slice if successful, null otherwise.
     */
    public static Cursive<ByteBuffer> slice(int length) {
        return buffer -> {
            if (buffer.remaining() < length) {
                return null;
            }
            int originalLimit = buffer.limit();
            buffer.limit(buffer.position() + length);
            ByteBuffer slice = buffer.slice();
            buffer.limit(originalLimit); // Restore original limit
            buffer.position(buffer.position() + length); // Advance original buffer's position
            return slice;
        };
    }

    /**
     * Creates a parser that duplicates the current ByteBuffer and returns the duplicate.
     * The position, limit, and mark of the new buffer will be independent.
     *
     * @return A Cursive parser that returns a duplicated ByteBuffer.
     */
    public static Cursive<ByteBuffer> duplicate() {
        return ByteBuffer::duplicate;
    }

    /**
     * Creates a parser that returns the current position of the ByteBuffer.
     * Does not advance the buffer's position.
     *
     * @return A Cursive parser that returns the current position as an Integer.
     */
    public static Cursive<Integer> positionP() {
        return ByteBuffer::position;
    }

    /**
     * Creates a parser that returns the current limit of the ByteBuffer.
     * Does not advance the buffer's position.
     *
     * @return A Cursive parser that returns the current limit as an Integer.
     */
    public static Cursive<Integer> limitP() {
        return ByteBuffer::limit;
    }

    /**
     * Creates a parser that reads a fixed-length string (UTF-8).
     *
     * @param length The number of bytes to read for the string.
     * @return A Cursive parser that returns the decoded String.
     */
    public static Cursive<String> string(int length) {
        return buffer -> {
            if (buffer.remaining() < length) {
                return null;
            }
            byte[] bytes = new byte[length];
            int originalPos = buffer.position();
            buffer.get(bytes);
            return new String(bytes, StandardCharsets.UTF_8);
        };
    }
}
