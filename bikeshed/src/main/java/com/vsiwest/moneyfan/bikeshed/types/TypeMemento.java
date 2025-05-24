package com.vsiwest.moneyfan.bikeshed.types;

/**
 * Represents a memento (snapshot) of a data type, primarily for serialization/deserialization purposes.
 * It provides information about the type's fixed size in a network/storage format.
 */
public interface TypeMemento {

    /**
     * The fixed network size (in bytes) of this type, or {@code null} if variable-length.
     * For ISAM alignment, fixed-size types are crucial.
     *
     * @return The size in bytes, or null for variable-length types.
     */
    Integer networkSize();
}
