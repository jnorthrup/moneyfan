package borg.trikeshed.nio;

import java.nio.ByteBuffer;

public class BooleanFieldSerializer implements ByteFieldSerializer<Boolean> {
    @Override
    public void serialize(ByteBuffer recordBuffer, int fieldOffsetInRecord, Boolean value, int fieldLength) {
        recordBuffer.put(fieldOffsetInRecord, (byte) ((value != null && value) ? 1 : 0));
    }
}
