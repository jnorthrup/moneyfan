package com.moneyfan.dsel.core;

/**
 * Represents the type information for data within the DSEL.
 * This can be extended to include more complex type details.
 */
public interface TypeMemento {

    /**
     * Basic types supported by the DSEL.
     */
    enum Basic implements TypeMemento {
        BOOLEAN,
        BYTE,
        SHORT,
        INTEGER,
        LONG,
        FLOAT,
        DOUBLE,
        CHAR,
        STRING,
        OBJECT, // Generic object type
        JOIN,   // Indicates the type is a Join itself
        // For more complex structures, the specific Join parameterization would be needed,
        // or dedicated TypeMemento implementations.
        CUSTOM // For user-defined types or complex nested structures
    }
}
