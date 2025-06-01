package borg.trikeshed.nio;

import java.nio.ByteBuffer;

public interface ByteFieldSerializer<T> {
    void serialize(ByteBuffer recordBuffer, int fieldOffsetInRecord, T value, int fieldLength);
}
