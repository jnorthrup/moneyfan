package borg.trikeshed.nio;

import java.nio.ByteBuffer;

public class IntegerFieldSerializer implements ByteFieldSerializer<Integer> {
    @Override
    public void serialize(ByteBuffer recordBuffer, int fieldOffsetInRecord, Integer value, int fieldLength) {
        recordBuffer.putInt(fieldOffsetInRecord, value == null ? 0 : value);
    }
}
