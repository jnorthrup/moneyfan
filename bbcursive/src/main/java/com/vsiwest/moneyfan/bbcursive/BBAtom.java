package com.vsiwest.moneyfan.bbcursive;

import com.vsiwest.moneyfan.bbcursive.core.Cursive;

import java.nio.ByteBuffer;
import java.util.NoSuchElementException;
import java.util.Objects;

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

    public static Cursive<Short> readShort() {
        return buffer -> {
            if (buffer.remaining() < 2) {
                throw new NoSuchElementException("Not enough bytes to read a short.");
            }
            return buffer.getShort();
        };
    }

    public static Cursive<Integer> readInt() {
        return buffer -> {
            if (buffer.remaining() < 4) {
                throw new NoSuchElementException("Not enough bytes to read an int.");
            }
            return buffer.getInt();
        };
    }

    public static Cursive<Long> readLong() {
        return buffer -> {
            if (buffer.remaining() < 8) {
                throw new NoSuchElementException("Not enough bytes to read a long.");
            }
            return buffer.getLong();
        };
    }

    public static Cursive<Float> readFloat() {
        return buffer -> {
            if (buffer.remaining() < 4) {
                throw new NoSuchElementException("Not enough bytes to read a float.");
            }
            return buffer.getFloat();
        };
    }

    public static Cursive<Double> readDouble() {
        return buffer -> {
            if (buffer.remaining() < 8) {
                throw new NoSuchElementException("Not enough bytes to read a double.");
            }
            return buffer.getDouble();
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
}
