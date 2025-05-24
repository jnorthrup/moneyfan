package com.vsiwest.bikeshed.type;

/**
 * Defines the contract for DSEL type information.
 * Implementations will specify serialization/deserialization strategies
 * and fixed sizes where applicable.
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
