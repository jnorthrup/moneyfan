package com.vsiwest.bikeshed.types;

import org.jetbrains.annotations.Nullable;

/**
 * Represents a type's metadata, particularly its fixed size in a network/storage context.
 * This is crucial for ISAM (Indexed Sequential Access Method) where record layouts are fixed.
 */
public interface TypeMemento {

    /**
     * The fixed network size (in bytes) of this type, or {@code null} if variable-length.
     * For ISAM alignment, fixed-size types are crucial.
     *
     * @return The size in bytes, or null for variable-length types.
     */
    @Nullable Integer networkSize();
}
