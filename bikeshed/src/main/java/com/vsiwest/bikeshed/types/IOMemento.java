package com.vsiwest.bikeshed.types;

import com.vsiwest.bbcursive.BBAtom;
import com.vsiwest.bbcursive.BBCombinator;
import com.vsiwest.bbcursive.core.Cursive;
import com.vsiwest.bbcursive.core.ParseResult;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.LocalDate;
import java.util.List;
import java.util.Objects;
import java.util.function.BiConsumer;
import java.util.function.Function;

public enum IOMemento implements TypeMemento {
    IoBoolean(Byte.BYTES) {
        @Override
        public Cursive<Object> getDecoder(int size) { return BBAtom.readByte().map(b -> (Object) (b != 0)); }
        @Override
        public BiConsumer<Object, ByteBuffer> getEncoder(int size) { return (val, buf) -> buf.put((byte) (((Boolean) val) ? 1 : 0)); }
    },
    IoByte(Byte.BYTES) {
        @Override public Cursive<Object> getDecoder(int size) { return BBAtom.readByte().map(b -> (Object)b); }
        @Override public BiConsumer<Object, ByteBuffer> getEncoder(int size) { return (val, buf) -> buf.put((Byte) val); }
    },
    IoShort(Short.BYTES) {
        @Override public Cursive<Object> getDecoder(int size) { return BBAtom.readShort().map(s -> (Object)s); }
        @Override public BiConsumer<Object, ByteBuffer> getEncoder(int size) { return (val, buf) -> buf.putShort((Short) val); }
    },
    IoInt(Integer.BYTES) {
        @Override public Cursive<Object> getDecoder(int size) { return BBAtom.readInt().map(i -> (Object)i); }
        @Override public BiConsumer<Object, ByteBuffer> getEncoder(int size) { return (val, buf) -> buf.putInt((Integer) val); }
    },
    IoLong(Long.BYTES) {
        @Override public Cursive<Object> getDecoder(int size) { return BBAtom.readLong().map(l -> (Object)l); }
        @Override public BiConsumer<Object, ByteBuffer> getEncoder(int size) { return (val, buf) -> buf.putLong((Long) val); }
    },
    IoFloat(Float.BYTES) {
        @Override public Cursive<Object> getDecoder(int size) { return BBAtom.readFloat().map(f -> (Object)f); }
        @Override public BiConsumer<Object, ByteBuffer> getEncoder(int size) { return (val, buf) -> buf.putFloat((Float) val); }
    },
    IoDouble(Double.BYTES) {
        @Override public Cursive<Object> getDecoder(int size) { return BBAtom.readDouble().map(d -> (Object)d); }
        @Override public BiConsumer<Object, ByteBuffer> getEncoder(int size) { return (val, buf) -> buf.putDouble((Double) val); }
    },
    IoChar(Character.BYTES) {
        @Override public Cursive<Object> getDecoder(int size) { return BBAtom.readChar().map(c -> (Object)c); }
        @Override public BiConsumer<Object, ByteBuffer> getEncoder(int size) { return (val, buf) -> buf.putChar((Character) val); }
    },
    IoInstant(Long.BYTES + Integer.BYTES) {
        @Override public Cursive<Object> getDecoder(int size) {
            return BBCombinator.sequence(BBAtom.readLong(), BBAtom.readInt())
                    .map(list -> {
                        Long epochSecond = (Long) list.get(0);
                        Integer nano = (Integer) list.get(1);
                        return (Object) Instant.ofEpochSecond(epochSecond, nano);
                    });
        }
        @Override public BiConsumer<Object, ByteBuffer> getEncoder(int size) {
            return (val, buf) -> {
                Instant inst = (Instant) val;
                buf.putLong(inst.getEpochSecond());
                buf.putInt(inst.getNano());
            };
        }
    },
    IoLocalDate(Long.BYTES) {
        @Override public Cursive<Object> getDecoder(int size) { return BBAtom.readLong().map(epochDay -> (Object) LocalDate.ofEpochDay(epochDay)); }
        @Override public BiConsumer<Object, ByteBuffer> getEncoder(int size) {
            return (val, buf) -> {
                LocalDate date = (LocalDate) val;
                buf.putLong(date.toEpochDay());
            };
        }
    },
    IoString(null) {
        @Override public Cursive<Object> getDecoder(int size) { return BBAtom.readString(size).map(s -> (Object)s); }
        @Override public BiConsumer<Object, ByteBuffer> getEncoder(int size) {
            return (val, buf) -> {
                byte[] strBytes = ((String) val).getBytes(StandardCharsets.UTF_8);
                int lenToCopy = Math.min(strBytes.length, size);
                buf.put(strBytes, 0, lenToCopy);
                for (int i = lenToCopy; i < size; i++) buf.put((byte) 0); // Pad with nulls
            };
        }
    },
    IoByteArray(null) {
        @Override public Cursive<Object> getDecoder(int size) {
            return BBAtom.readSlice(size).map(bb -> {
                byte[] bytes = new byte[bb.remaining()];
                bb.get(bytes);
                return (Object) bytes;
            });
        }
        @Override public BiConsumer<Object, ByteBuffer> getEncoder(int size) {
            return (val, buf) -> {
                byte[] bytes = (byte[]) val;
                int lenToCopy = Math.min(bytes.length, size);
                buf.put(bytes, 0, lenToCopy);
                for (int i = lenToCopy; i < size; i++) buf.put((byte) 0); // Pad with nulls
            };
        }
    };

    private final Integer networkSize;

    IOMemento(@Nullable Integer networkSize) {
        this.networkSize = networkSize;
    }

    @Override
    public Integer networkSize() {
        return networkSize;
    }

    public abstract Cursive<Object> getDecoder(int size);
    public abstract BiConsumer<Object, ByteBuffer> getEncoder(int size);

    public static @NotNull IOMemento fromTypeName(@NotNull String typeName) {
        String normalizedTypeName = typeName.split("\\(")[0].trim();
        for (IOMemento memento : values()) {
            if (memento.name().equalsIgnoreCase(normalizedTypeName)) {
                return memento;
            }
        }
        throw new IllegalArgumentException("Unknown IOMemento type name: " + typeName);
    }
}
