package com.example.bikeshed.isam;

import com.example.bikeshed.types.ColumnMeta;
import com.example.bikeshed.types.IOMemento;
import com.example.bikeshed.types.TypeMemento;

import java.nio.ByteBuffer;
import java.util.function.Function;

/**
 * Extends ColumnMeta to include byte offsets and direct decoder/encoder functions
 * for ISAM fixed-format records.
 */
public class RecordMeta extends ColumnMeta {
    private final int begin;
    private final int end;
    private final Function<ByteBuffer, Object> decoder;
    private final IOMemento.BiConsumer<Object, ByteBuffer> encoder;

    public RecordMeta(String name, IOMemento type, int begin, int end,
                      Function<ByteBuffer, Object> decoder,
                      IOMemento.BiConsumer<Object, ByteBuffer> encoder) {
        super(name, type);
        this.begin = begin;
        this.end = end;
        this.decoder = decoder;
        this.encoder = encoder;
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

    public IOMemento.BiConsumer<Object, ByteBuffer> getEncoder() {
        return encoder;
    }

    public int getLength() {
        return end - begin;
    }

    @Override
    public String toString() {
        return "RecordMeta{" +
               "name='" + getName() + '\'' +
               ", type=" + getType() +
               ", begin=" + begin +
               ", end=" + end +
               ", length=" + getLength() +
               '}';
    }
}
