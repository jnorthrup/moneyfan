package borg.trikeshed.nio;

import java.nio.ByteBuffer;

public class DoubleFieldSerializer implements ByteFieldSerializer<Double> {
    @Override
    public void serialize(ByteBuffer recordBuffer, int fieldOffsetInRecord, Double value, int fieldLength) {
        recordBuffer.putDouble(fieldOffsetInRecord, value == null ? 0.0 : value);
    }
}
