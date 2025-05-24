package com.vsiwest.bikeshed.types;

import java.nio.ByteBuffer;
import java.time.Instant;
import java.time.LocalDate;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;

public enum IOMemento implements TypeMemento {
    IoBoolean(1),
    IoByte(1),
    IoShort(2),
    IoInt(4),
    IoLong(8),
    IoFloat(4),
    IoDouble(8),
    IoChar(2), // Assuming 2 bytes for char (UTF-16 standard for Java)
    IoLocalDate(8) { // epoch days (long)
        @Override
        public Function<ByteBuffer, Object> createDecoder(int size) {
            return buffer -> {