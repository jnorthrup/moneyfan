package com.yourdomain.bbcursive;

import java.nio.ByteBuffer;
import java.util.Objects;
import java.util.NoSuchElementException;

public enum BBAtom {
    ; // No specific instances needed, just static utility methods.

    public static Cursive<Byte> readByte() {
        return buffer -> {
            if (!buffer.hasRemaining()) {
                throw new NoSuchElementException("Buffer exhausted.");
            }
            return buffer.get();
        };
    }

    public static Cursive<Byte> matchByte(byte expectedByte) {
        return buffer -> {
            Objects.requireNonNull(buffer, "ByteBuffer must not be null");
            byte actualByte = readByte().apply(buffer); // Use readByte() parser
            if (actualByte != expectedByte) {
                throw new IllegalArgumentException("Expected byte " + expectedByte + ", but got " + actualByte);
            }
            return actualByte;
        };
    }

    public static Cursive<ByteBuffer> slice(int length) {
        return buffer -> {
            Objects.requireNonNull(buffer, "ByteBuffer must not be null");
            if (buffer.remaining() < length) {
                throw new NoSuchElementException("Not enough bytes to slice (" + length + " requested, " + buffer.remaining() + " available).");
            }
            int originalLimit = buffer.limit();
            int originalPosition = buffer.position();
            buffer.limit(originalPosition + length); // Set limit for the slice
            ByteBuffer slice = buffer.slice();       // Create slice from current position to new limit
            buffer.limit(originalLimit);             // Restore original limit
            buffer.position(originalPosition + length); // Advance original buffer's position
            return slice;
        };
    }

    public static void resetBufferPosition(ByteBuffer buffer) {
        Objects.requireNonNull(buffer, "ByteBuffer must not be null");
        try {
            buffer.reset();
        } catch (java.nio.InvalidMarkException e) {
            buffer.position(0); // If mark is not set, reset to start
        }
    }
}
