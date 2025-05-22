package com.moneyfan.core;

/**
 * Record for column type and name.
 */
public record Scalar(IOMemento type, String name, int stringLength) {
    
    public Scalar(IOMemento type, String name) {
        this(type, name, -1);
    }
    
    public static Scalar of(IOMemento type, String name) {
        return new Scalar(type, name);
    }
    
    public static Scalar stringOf(String name, int length) {
        return new Scalar(IOMemento.IO_STRING_FIXED, name, length);
    }
    
    public boolean isStringType() {
        return type == IOMemento.IO_STRING_FIXED;
    }
}