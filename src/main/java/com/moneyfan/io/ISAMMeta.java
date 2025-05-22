package com.moneyfan.io;

import com.moneyfan.core.IOMemento;
import com.moneyfan.core.Scalar;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Record to represent/parse/write ISAM metadata.
 */
public record ISAMMeta(List<Scalar> scalars, List<Integer> offsets, int recordLength) {
    
    public static ISAMMeta fromScalars(List<Scalar> scalars) {
        List<Integer> offsets = new ArrayList<>(scalars.size());
        int currentOffset = 0;
        
        for (Scalar scalar : scalars) {
            offsets.add(currentOffset);
            
            if (scalar.type() == IOMemento.IO_STRING_FIXED) {
                currentOffset += scalar.stringLength();
            } else {
                currentOffset += scalar.type().getSize();
            }
        }
        
        return new ISAMMeta(scalars, offsets, currentOffset);
    }
    
    public static ISAMMeta fromFile(Path metaPath) throws IOException {
        List<String> lines = Files.readAllLines(metaPath);
        int recordLength = Integer.parseInt(lines.get(0));
        
        List<Scalar> scalars = new ArrayList<>();
        List<Integer> offsets = new ArrayList<>();
        
        for (int i = 1; i < lines.size(); i++) {
            String[] parts = lines.get(i).split(",");
            String name = parts[0];
            String type = parts[1];
            int offset = Integer.parseInt(parts[2]);
            offsets.add(offset);
            
            if (type.startsWith("string:")) {
                int stringLength = Integer.parseInt(type.substring(7));
                scalars.add(Scalar.stringOf(name, stringLength));
            } else {
                IOMemento ioType = IOMemento.valueOf(type);
                scalars.add(Scalar.of(ioType, name));
            }
        }
        
        return new ISAMMeta(scalars, offsets, recordLength);
    }
    
    public void writeToFile(Path metaPath) throws IOException {
        List<String> lines = new ArrayList<>();
        lines.add(String.valueOf(recordLength));
        
        for (int i = 0; i < scalars.size(); i++) {
            Scalar scalar = scalars.get(i);
            int offset = offsets.get(i);
            
            String typeStr = scalar.type().name();
            if (scalar.type() == IOMemento.IO_STRING_FIXED) {
                typeStr = "string:" + scalar.stringLength();
            }
            
            lines.add(scalar.name() + "," + typeStr + "," + offset);
        }
        
        Files.write(metaPath, lines);
    }
    
    public int getOffsetForColumn(int columnIndex) {
        return offsets.get(columnIndex);
    }
    
    public Scalar getScalarForColumn(int columnIndex) {
        return scalars.get(columnIndex);
    }
    
    public int getColumnCount() {
        return scalars.size();
    }
}