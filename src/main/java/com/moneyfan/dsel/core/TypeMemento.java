package com.moneyfan.dsel.core;

import java.util.Objects;

public interface TypeMemento {
    String getTypeName();
    int getFixedSize();

    enum Basic implements TypeMemento {
        BOOLEAN("Boolean", 1), BYTE("Byte", 1), SHORT("Short", 2), INTEGER("Integer", 4), LONG("Long", 8),
        FLOAT("Float", 4), DOUBLE("Double", 8), CHAR("Char", 2), STRING("String", -1), BINARY_BLOB("BinaryBlob", -1),
        OBJECT("Object", -1), JOIN("Join", -1), SERIES("Series", -1), ROWVEC("RowVec", -1), CURSOR("Cursor", -1),
        TWIN("Twin", -1), CUSTOM("Custom", -1); // CUSTOM type can represent dynamically sized types

        private final String tn;
        private final int fs;

        Basic(String tn, int fs) {
            this.tn = tn;
            this.fs = fs;
        }

        @Override
        public String getTypeName() { return tn; }
        @Override
        public int getFixedSize() { return fs; }

        public static TypeMemento fromTypeName(String name) {
            for (Basic b : values()) {
                if (b.getTypeName().equals(name)) return b;
            }
            // If not a basic type, it might be a custom-sized string, try to parse it
            if (name.startsWith(CUSTOM_STRING_PREFIX)) {
                try {
                    int size = Integer.parseInt(name.substring(CUSTOM_STRING_PREFIX.length()));
                    return customString(size);
                } catch (NumberFormatException e) {
                    // Fall through to throw exception if it's not a valid custom string format either
                }
            }
            throw new IllegalArgumentException("Unknown TypeMemento name: " + name);
        }
    }

    // New concrete implementation for custom-sized types (e.g., Strings with a specific length)
    class CustomType implements TypeMemento {
        private final String typeName;
        private final int fixedSize;

        // Make the constructor public
        public CustomType(String typeName, int fixedSize) {
            this.typeName = Objects.requireNonNull(typeName);
            this.fixedSize = fixedSize;
        }

        @Override
        public String getTypeName() {
            return typeName;
        }

        @Override
        public int getFixedSize() {
            return fixedSize;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            CustomType that = (CustomType) o;
            return fixedSize == that.fixedSize && typeName.equals(that.typeName);
        }

        @Override
        public int hashCode() {
            return Objects.hash(typeName, fixedSize);
        }

        @Override
        public String toString() {
            return "CustomType{" +
                   "typeName='" + typeName + '\'' +
                   ", fixedSize=" + fixedSize +
                   '}';
        }
    }

    // Helper for custom string types
    String CUSTOM_STRING_PREFIX = "String_Fixed_";

    static TypeMemento customString(int fixedSize) {
        if (fixedSize <= 0) throw new IllegalArgumentException("Fixed size for custom string must be positive.");
        return new CustomType(CUSTOM_STRING_PREFIX + fixedSize, fixedSize);
    }
}
