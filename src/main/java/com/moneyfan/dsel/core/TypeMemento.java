package com.moneyfan.dsel.core;
public interface TypeMemento {
    String getTypeName(); int getFixedSize();
    enum Basic implements TypeMemento {
        BOOLEAN("Boolean", 1), BYTE("Byte", 1), SHORT("Short", 2), INTEGER("Integer", 4), LONG("Long", 8),
        FLOAT("Float", 4), DOUBLE("Double", 8), CHAR("Char", 2), STRING("String", -1), BINARY_BLOB("BinaryBlob", -1),
        OBJECT("Object", -1), JOIN("Join", -1), SERIES("Series", -1), ROWVEC("RowVec", -1), CURSOR("Cursor", -1),
        TWIN("Twin", -1), CUSTOM("Custom", -1);
        private final String tn; private final int fs;
        Basic(String tn, int fs) { this.tn = tn; this.fs = fs; }
        @Override public String getTypeName() { return tn; } @Override public int getFixedSize() { return fs; }
        public static TypeMemento fromTypeName(String name) {
            for (Basic b : values()) if (b.getTypeName().equals(name)) return b;
            throw new IllegalArgumentException("Unknown TypeMemento name: " + name);
        }
    }
}