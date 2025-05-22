package com.moneyfan.core;

/**
 * Scalar describes metadata for a column: its IO type and logical name.
 * Immutable value object.
 */
public record Scalar(IOMemento type, String name) {
    public static Scalar of(IOMemento type, String name) {
        return new Scalar(type, name);
    }
}