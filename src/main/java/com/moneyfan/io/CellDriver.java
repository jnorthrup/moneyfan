package com.moneyfan.io;

import java.nio.ByteBuffer;

/**
 * Strategy interface for reading/writing cell values from a binary buffer.
 * @param <T> Java type of cell value
 */
public interface CellDriver<T> {

    /**
     * Reads a value from buffer at given offset.
     */
    T read(ByteBuffer buffer, int offset);

    /**
     * Writes value into buffer at given offset.
     */
    void write(ByteBuffer buffer, int offset, T value);

    /**
     * @return fixed size in bytes for this driver. For variable length types, return -1.
     */
    int size();
}