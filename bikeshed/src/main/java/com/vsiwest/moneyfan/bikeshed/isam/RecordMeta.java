package com.vsiwest.moneyfan.bikeshed.isam;

import com.vsiwest.moneyfan.bikeshed.types.ColumnMeta;
import com.vsiwest.moneyfan.bikeshed.types.IOMemento; // Assuming IOMemento is in this package
import com.vsiwest.moneyfan.bikeshed.types.TypeMemento;

import java.nio.ByteBuffer;
import java.util.function.BiConsumer;
import java.util.function.Function;
import java.util.Objects;

/**
 * Extends ColumnMeta to include byte offsets and direct decoder/encoder functions
 * for ISAM fixed-format records.
 */
public class RecordMeta extends ColumnMeta {
    private final int begin;
    private final int end;
    private final Function<ByteBuffer, Object> decoder;
    private final BiConsumer<Object, ByteBuffer> encoder;

    public RecordMeta(String name, IOMemento type, int begin, int end,
                      Function<ByteBuffer, Object> decoder,
                      BiConsumer<Object, ByteBuffer> encoder) {
        super(name, type);
        this.begin = begin;
        this.end = end;
        this.decoder = Objects.requireNonNull(decoder, "decoder must not be null");
        this.encoder = Objects.requireNonNull(encoder, "encoder must not be null");
    }

    public int getBegin() {
        return begin;
    }

    public int getEnd() {
        return end;
    }

    public Function<ByteBuffer, Object> getDecoder() {
        return decoder;
    }

    public BiConsumer<Object, ByteBuffer> getEncoder() {
        return encoder;
    }

    public int getLength() {
        return end - begin;
    }

    @Override
    public String toString() {
        return "RecordMeta{" +
               "name='" + name() + '\'' +
               ", type=" + type() +
               ", begin=" + begin +
               ", end=" + end +
               ", length=" + getLength() +
               '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        if (!super.equals(o)) return false; // Check equality of superclass (ColumnMeta) fields
        RecordMeta that = (RecordMeta) o;
        return begin == that.begin &&
               end == that.end &&
               Objects.equals(decoder, that.decoder) && // Note: Function/BiConsumer equality is tricky, often by reference
               Objects.equals(encoder, that.encoder);
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.hashCode(), begin, end, decoder, encoder);
    }
}
