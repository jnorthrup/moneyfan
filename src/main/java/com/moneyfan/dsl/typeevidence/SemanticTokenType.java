package com.moneyfan.dsl.typeevidence;

/**
 * Enum representing the semantic types of data fields.
 * This enum itself is a "bag of code elements" (the types).
 */
public enum SemanticTokenType {
    STRING,
    INTEGER,
    LONG,
    DOUBLE,
    BOOLEAN,
    BIG_DECIMAL,
    DATE_TIME, // e.g., java.time.LocalDateTime or Instant
    BYTE_ARRAY,
    OBJECT,    // Generic Java object
    ROW,       // For nested Row structures
    LIST,      // For lists of a certain type
    MAP,       // For key-value maps
    UNKNOWN;

    public boolean isNumeric() {
        return this == INTEGER || this == LONG || this == DOUBLE || this == BIG_DECIMAL;
    }
}
