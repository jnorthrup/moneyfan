package com.moneyfan.core;

/**
 * Column metadata describing the value type and column name.
 * Immutable by design.
 *
 * @param type  the {@link IOMemento} representing how the value is stored
 * @param name  the logical column name (human readable)
 */
public record Scalar(IOMemento type, String name) {
}