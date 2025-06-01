package borg.trikeshed.nio;

import java.nio.ByteBuffer;

public class LongFieldSerializer implements ByteFieldSerializer<Long> {
    @Override
    public void serialize(ByteBuffer recordBuffer, int fieldOffsetInRecord, Long value, int fieldLength) {
        recordBuffer.putLong(fieldOffsetInRecord, value == null ? 0L : value);
    }
}
