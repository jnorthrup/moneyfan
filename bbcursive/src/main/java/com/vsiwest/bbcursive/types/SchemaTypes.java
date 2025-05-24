package com.vsiwest.bbcursive.types;

import com.vsiwest.bbcursive.BBAtom;
import com.vsiwest.bbcursive.Cursive;
import com.vsiwest.bbcursive.core.Join;
import org.jetbrains.annotations.Nullable;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.LocalDate;
import java.util.function.BiConsumer;

public interface TypeMemento {}
public interface ColumnMeta extends Join<String, TypeMemento> {}
record ColumnMetaImpl(String name, TypeMemento type) implements ColumnMeta { @Override public String first() { return name; } @Override public TypeMemento second() { return type; } }

public enum IOMemento implements TypeMemento {
    IoBoolean(Byte.BYTES) { @Override public Cursive<Object> decoder(int l) { return BBAtom.byteP().map(b -> (Object)(b != null && b != 0)); } @Override public BiConsumer<Object, ByteBuffer> encoder(int l) { return (v, b) -> b.put((byte)(((Boolean)v) ? 1 : 0)); } },
    IoByte(Byte.BYTES) { @Override public Cursive<Object> decoder(int l) { return BBAtom.byteP().map(o -> o); } @Override public BiConsumer<Object, ByteBuffer> encoder(int l) { return (v, b) -> b.put((Byte)v); } },
    IoShort(Short.BYTES) { @Override public Cursive<Object> decoder(int l) { return BBAtom.shortP().map(o -> o); } @Override public BiConsumer<Object, ByteBuffer> encoder(int l) { return (v, b) -> b.putShort((Short)v); } },
    IoInt(Integer.BYTES) { @Override public Cursive<Object> decoder(int l) { return BBAtom.intP().map(o -> o); } @Override public BiConsumer<Object, ByteBuffer> encoder(int l) { return (v, b) -> b.putInt((Integer)v); } },
    IoLong(Long.BYTES) { @Override public Cursive<Object> decoder(int l) { return BBAtom.longP().map(o -> o); } @Override public BiConsumer<Object, ByteBuffer> encoder(int l) { return (v, b) -> b.putLong((Long)v); } },
    IoFloat(Float.BYTES) { @Override public Cursive<Object> decoder(int l) { return BBAtom.floatP().map(o -> o); } @Override public BiConsumer<Object, ByteBuffer> encoder(int l) { return (v, b) -> b.putFloat((Float)v); } },
    IoDouble(Double.BYTES) { @Override public Cursive<Object> decoder(int l) { return BBAtom.doubleP().map(o -> o); } @Override public BiConsumer<Object, ByteBuffer> encoder(int l) { return (v, b) -> b.putDouble((Double)v); } },
    IoChar(Character.BYTES) { @Override public Cursive<Object> decoder(int l) { return BBAtom.charP().map(o -> o); } @Override public BiConsumer<Object, ByteBuffer> encoder(int l) { return (v, b) -> b.putChar((Character)v); } },
    IoInstant(Long.BYTES + Integer.BYTES) { @Override public Cursive<Object> decoder(int l) { return buf -> { Long s = BBAtom.longP().apply(buf); Integer n = BBAtom.intP().apply(buf); return s != null && n != null ? Instant.ofEpochSecond(s, n) : null; }; } @Override public BiConsumer<Object, ByteBuffer> encoder(int l) { return (v,b) -> { Instant i=(Instant)v; b.putLong(i.getEpochSecond()); b.putInt(i.getNano()); }; } },
    IoLocalDate(Long.BYTES) { @Override public Cursive<Object> decoder(int l) { return BBAtom.longP().map(LocalDate::ofEpochDay); } @Override public BiConsumer<Object, ByteBuffer> encoder(int l) { return (v,b) -> b.putLong(((LocalDate)v).toEpochDay()); }; },
    IoString(null) { @Override public Cursive<Object> decoder(int l) { return BBAtom.stringP(l).map(o->o); } @Override public BiConsumer<Object, ByteBuffer> encoder(int l) { return (v,b) -> { byte[] s = ((String)v).getBytes(StandardCharsets.UTF_8); int len = Math.min(s.length, l); b.put(s, 0, len); for(int i=len; i<l; i++) b.put((byte)0); }; } },
    IoByteArray(null) { @Override public Cursive<Object> decoder(int l) { return BBAtom.bytesP(l).map(o->o); } @Override public BiConsumer<Object, ByteBuffer> encoder(int l) { return (v,b) -> { byte[] s = (byte[])v; int len = Math.min(s.length, l); b.put(s, 0, len); for(int i=len; i<l; i++) b.put((byte)0); }; } };
    private final @Nullable Integer networkSize;
    IOMemento(@Nullable Integer networkSize) { this.networkSize = networkSize; }
    public @Nullable Integer getNetworkSize() { return networkSize; }
    public abstract Cursive<Object> decoder(int length);
    public abstract BiConsumer<Object, ByteBuffer> encoder(int length);
}
