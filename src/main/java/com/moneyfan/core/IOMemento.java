package com.moneyfan.core;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneOffset;

/**
 * {@code IOMemento} describes how a given logical Java type is represented in the binary
 * columnar format.  It knows the <em>wire size</em> (number of bytes on disk/network) and
 * provides minimal (non-allocating) helpers for reading/writing values from / to a
 * {@link ByteBuffer}.  The design purposefully keeps the API surface small; specialised
 * codecs or higher-level <em>CellDriver</em>s can be built on top later.
 */
public enum IOMemento {

    BOOLEAN(1, Boolean.class) {
        @Override
        public Object read(ByteBuffer buf, int fixedLen) {
            return buf.get() != 0;
        }
        @Override
        public void write(ByteBuffer buf, Object value, int fixedLen) {
            buf.put((byte) ((Boolean) value ? 1 : 0));
        }
    },
    BYTE(1, Byte.class) {
        @Override
        public Object read(ByteBuffer buf, int fixedLen) {return buf.get();}
        @Override
        public void write(ByteBuffer buf, Object value, int fixedLen) {buf.put((Byte) value);} },
    INT(4, Integer.class) {
        @Override
        public Object read(ByteBuffer buf, int fixedLen) {return buf.getInt();}
        @Override
        public void write(ByteBuffer buf, Object value, int fixedLen) {buf.putInt((Integer) value);} },
    LONG(8, Long.class) {
        @Override
        public Object read(ByteBuffer buf, int fixedLen) {return buf.getLong();}
        @Override
        public void write(ByteBuffer buf, Object value, int fixedLen) {buf.putLong((Long) value);} },
    FLOAT(4, Float.class) {
        @Override
        public Object read(ByteBuffer buf, int fixedLen) {return buf.getFloat();}
        @Override
        public void write(ByteBuffer buf, Object value, int fixedLen) {buf.putFloat((Float) value);} },
    DOUBLE(8, Double.class) {
        @Override
        public Object read(ByteBuffer buf, int fixedLen) {return buf.getDouble();}
        @Override
        public void write(ByteBuffer buf, Object value, int fixedLen) {buf.putDouble((Double) value);} },
    /**
     * Stored as number of days since <code>1970-01-01</code> (signed 64-bit).
     */
    LOCAL_DATE(8, LocalDate.class) {
        @Override
        public Object read(ByteBuffer buf, int fixedLen) {
            long epochDay = buf.getLong();
            return LocalDate.ofEpochDay(epochDay);
        }
        @Override
        public void write(ByteBuffer buf, Object value, int fixedLen) {
            long epochDay = ((LocalDate) value).toEpochDay();
            buf.putLong(epochDay);
        }
    },
    /**
     * Stored as <code>epochSecond (long)</code> followed by <code>nano (int)</code>.
     */
    INSTANT(12, Instant.class) {
        @Override
        public Object read(ByteBuffer buf, int fixedLen) {
            long epochSecond = buf.getLong();
            int nano = buf.getInt();
            return Instant.ofEpochSecond(epochSecond, nano);
        }
        @Override
        public void write(ByteBuffer buf, Object value, int fixedLen) {
            Instant inst = (Instant) value;
            buf.putLong(inst.getEpochSecond());
            buf.putInt(inst.getNano());
        }
    },
    /**
     * UTF-8 encoded string with fixed byte length.  Any unused space is zero-filled.
     * {@code fixedLen} parameter must be &gt; 0.
     */
    STRING_FIXED(-1, String.class) {
        @Override
        public Object read(ByteBuffer buf, int fixedLen) {
            if (fixedLen <= 0) throw new IllegalArgumentException("fixedLen required for STRING_FIXED");
            byte[] bytes = new byte[fixedLen];
            buf.get(bytes);
            int realLen = 0;
            while (realLen < bytes.length && bytes[realLen] != 0) realLen++;
            return new String(bytes, 0, realLen, StandardCharsets.UTF_8);
        }
        @Override
        public void write(ByteBuffer buf, Object value, int fixedLen) {
            if (fixedLen <= 0) throw new IllegalArgumentException("fixedLen required for STRING_FIXED");
            byte[] utf8 = ((String) value).getBytes(StandardCharsets.UTF_8);
            if (utf8.length > fixedLen) throw new IllegalArgumentException("String too long for fixed size");
            buf.put(utf8);
            // pad remaining
            for (int i = utf8.length; i < fixedLen; i++) buf.put((byte) 0);
        }
    },
    /**
     * Placeholder type representing an absent cell.  Wire size is zero.
     */
    NOTHING(0, Void.class) {
        @Override public Object read(ByteBuffer buf, int fixedLen) {return null;}
        @Override public void write(ByteBuffer buf, Object value, int fixedLen) {/* no-op */}
    };

    private final int networkSize;
    private final Class<?> javaType;

    IOMemento(int networkSize, Class<?> javaType) {
        this.networkSize = networkSize;
        this.javaType = javaType;
    }

    /**
     * @return the number of bytes the value occupies on disk / wire.  <br>
     * For {@link #STRING_FIXED} the size is supplied via <code>fixedLen</code> parameters.
     */
    public int networkSize() {return networkSize;}

    /**
     * @return logical Java type represented
     */
    public Class<?> javaType() {return javaType;}

    /**
     * Reads a value of this type from the {@code buffer} advancing its position accordingly.
     * @param buffer     byte buffer to read from
     * @param fixedLen   mandatory size for variable-length encodings (currently only STRING_FIXED)
     * @return decoded Java object (boxed where necessary)
     */
    public abstract Object read(ByteBuffer buffer, int fixedLen);

    /**
     * Writes {@code value} into {@code buffer} according to this memento.
     * @param buffer     target buffer
     * @param value      non-null value to write (null not supported at this layer)
     * @param fixedLen   same semantics as {@link #read(ByteBuffer, int)}
     */
    public abstract void write(ByteBuffer buffer, Object value, int fixedLen);

    /**
     * Convenience overload for fixed-size mementos.
     */
    public Object read(ByteBuffer buffer) {return read(buffer, networkSize);} // NOSONAR simple delegate

    /**
     * Convenience overload for fixed-size mementos.
     */
    public void write(ByteBuffer buffer, Object value) {write(buffer, value, networkSize);} // NOSONAR
}