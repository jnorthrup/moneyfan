package com.example.bikeshed.types;

/**
 * Base interface for defining data types within the DSEL.
 * Represents a "memento" of a type's properties.
 *
 * `networkSize` is crucial for ISAM alignment.
 * `null` indicates a variable-length type, requiring explicit length configuration (e.g., for `IoString`).
 */
public interface TypeMemento {
    /**
     * The fixed size in bytes for network serialization.
     * Returns `null` for variable-length types (e.g., String),
     * which require external length configuration (e.g., in `IsamMetaFileReader.write`).
     *
     * @return The fixed byte size for network serialization, or `null` for variable length.
     */
    Integer getNetworkSize();
}
