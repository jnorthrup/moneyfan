package com.moneyfan.io;

/**
 * Interface for reading/writing cell values.
 */
public interface CellDriver<Buffer, Value> {
    
    Value read(Buffer buffer, int offset);
    
    void write(Buffer buffer, int offset, Value value);
}