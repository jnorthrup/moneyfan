package com.vsiwest.bbcursive;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

public enum BBAtom implements Cursive<Object> {
    BYTE_PARSER { @Override public Byte apply(ByteBuffer b) { return b.hasRemaining() ? b.get() : null; } },
    SHORT_PARSER { @Override public Short apply(ByteBuffer b) { return b.remaining() >= 2 ? b.getShort() : null; } },
    INT_PARSER { @Override public Integer apply(ByteBuffer b) { return b.remaining() >= 4 ? b.getInt() : null; } },
    LONG_PARSER { @Override public Long apply(ByteBuffer b) { return b.remaining() >= 8 ? b.getLong() : null; } },
    FLOAT_PARSER { @Override public Float apply(ByteBuffer b) { return b.remaining() >= 4 ? b.getFloat() : null; } },
    DOUBLE_PARSER { @Override public Double apply(ByteBuffer b) { return b.remaining() >= 8 ? b.getDouble() : null; } },
    CHAR_PARSER { @Override public Character apply(ByteBuffer b) { return b.remaining() >= 2 ? b.getChar() : null; } };

    @SuppressWarnings("unchecked") public static <T> Cursive<T> atom(BBAtom atom) { return (Cursive<T>) atom; }
    public static Cursive<Byte> byteP() { return atom(BYTE_PARSER); }
    public static Cursive<Short> shortP() { return atom(SHORT_PARSER); }
    public static Cursive<Integer> intP() { return atom(INT_PARSER); }
    public static Cursive<Long> longP() { return atom(LONG_PARSER); }
    public static Cursive<Float> floatP() { return atom(FLOAT_PARSER); }
    public static Cursive<Double> doubleP() { return atom(DOUBLE_PARSER); }
    public static Cursive<Character> charP() { return atom(CHAR_PARSER); }
    public static Cursive<String> stringP(int len) { return b -> b.remaining() >= len ? new String(toArray(b, len), StandardCharsets.UTF_8) : null; }
    public static Cursive<byte[]> bytesP(int len) { return b -> b.remaining() >= len ? toArray(b, len) : null; }
    private static byte[] toArray(ByteBuffer b, int len) { byte[] dst = new byte[len]; b.get(dst); return dst; }
}
