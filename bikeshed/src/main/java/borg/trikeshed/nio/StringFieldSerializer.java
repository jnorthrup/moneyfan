package borg.trikeshed.nio;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

public class StringFieldSerializer implements ByteFieldSerializer<String> {
    @Override
    public void serialize(ByteBuffer recordBuffer, int fieldOffsetInRecord, String value, int fieldLength) {
        byte[] stringBytes = (value == null ? "" : value).getBytes(StandardCharsets.UTF_8);
        int bytesToWrite = Math.min(stringBytes.length, fieldLength);

        // Save current position, write, then restore. This ensures that even if this method
        // is called for multiple fields in arbitrary order within the same recordBuffer,
        // it writes to the correct place without relying on sequential calls.
        int originalPos = recordBuffer.position();
        recordBuffer.position(fieldOffsetInRecord);
        recordBuffer.put(stringBytes, 0, bytesToWrite);
        // Pad if necessary
        for (int i = bytesToWrite; i < fieldLength; i++) {
            recordBuffer.put((byte) 0); // Null padding
        }
        recordBuffer.position(originalPos); // Restore position
    }
}
