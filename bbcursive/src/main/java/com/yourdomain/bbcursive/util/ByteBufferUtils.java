package com.yourdomain.bbcursive.util;

import org.jetbrains.annotations.NotNull;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

/**
 * Utility methods for ByteBuffer manipulation, providing convenience functions
 * that might be commonly used but not strictly part of the core Cursive API.
 */
public enum ByteBufferUtils {
    ; // No instances

    /**
     * Creates a new ByteBuffer from a byte array.
     * @param bytes The byte array.
     * @return A new ByteBuffer wrapping the array.
     */
    public static @NotNull ByteBuffer wrap(@NotNull byte[] bytes) {
        return ByteBuffer.wrap(bytes);
    }

    /**
     * Creates a new direct ByteBuffer with the specified capacity.
     * @param capacity The capacity of the buffer.
     * @return A new direct ByteBuffer.
     */
    public static @NotNull ByteBuffer allocateDirect(int capacity) {
        return ByteBuffer.allocateDirect(capacity);
    }

    /**
     * Creates a new heap ByteBuffer with the specified capacity.
     * @param capacity The capacity of the buffer.
     * @return A new heap ByteBuffer.
     */
    public static @NotNull ByteBuffer allocate(int capacity) {
        return ByteBuffer.allocate(capacity);
    }

    /**
     * Converts a ByteBuffer to a UTF-8 String from its current position to its limit.
     * @param buffer The buffer to convert.
     * @return The String representation.
     */
    public static @NotNull String toString(@NotNull ByteBuffer buffer) {
        byte[] bytes = new byte[buffer.remaining()];
        int originalPosition = buffer.position();
        buffer.get(bytes);
        buffer.position(originalPosition); // Restore position
        return new String(bytes, StandardCharsets.UTF_8);
    }

    /**
     * Copies the remaining bytes from the source buffer to a new byte array.
     * The source buffer's position is advanced to its limit.
     * @param buffer The source buffer.
     * @return A new byte array containing the copied bytes.
     */
    public static @NotNull byte[] toByteArray(@NotNull ByteBuffer buffer) {
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes);
        return bytes;
    }

    /**
     * Sets the byte order for the given ByteBuffer.
     * @param buffer The buffer.
     * @param order The desired byte order.
     * @return The buffer with the new byte order applied.
     */
    public static @NotNull ByteBuffer withByteOrder(@NotNull ByteBuffer buffer, @NotNull ByteOrder order) {
        return buffer.order(order);
    }

    /**
     * Utility to create a duplicate buffer with independent position/limit, but shared content.
     * @param buffer The original buffer.
     * @return A duplicated buffer.
     */
    public static @NotNull ByteBuffer duplicate(@NotNull ByteBuffer buffer) {
        return buffer.duplicate();
    }
}
