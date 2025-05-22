package com.moneyfan.io;

import com.moneyfan.core.IOMemento;
import com.moneyfan.core.Scalar;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.StringJoiner;

/**
 * Simple metadata description for ISAM files.
 * Format (CSV-like):
 *   #recordLength=<int>
 *   columnName,type,fixedLength
 * For non-string-fixed types, fixedLength should be -1.
 */
public record ISAMMeta(List<Scalar> columns, List<Integer> fixedStringLengths, List<Integer> offsets, int recordLength) {

    public ISAMMeta {
        Objects.requireNonNull(columns, "columns");
        Objects.requireNonNull(fixedStringLengths, "fixedStringLengths");
        Objects.requireNonNull(offsets, "offsets");
        if(columns.size()!=fixedStringLengths.size() || columns.size()!=offsets.size())
            throw new IllegalArgumentException("All column lists must align in size");
    }

    public static ISAMMeta fromColumns(List<Scalar> cols, List<Integer> fixedLengths) {
        if(cols.size()!=fixedLengths.size())
            throw new IllegalArgumentException("fixed lengths list must match columns");
        List<Integer> offsets = new ArrayList<>(cols.size());
        int off = 0;
        for(int i=0;i<cols.size();i++) {
            offsets.add(off);
            IOMemento t = cols.get(i).type();
            if(t== IOMemento.IO_STRING_FIXED) {
                off += fixedLengths.get(i);
            } else {
                off += t.fixedSize();
            }
        }
        int recLen = off;
        return new ISAMMeta(List.copyOf(cols), List.copyOf(fixedLengths), List.copyOf(offsets), recLen);
    }

    public static ISAMMeta read(Path metaPath) throws IOException {
        try (BufferedReader br = Files.newBufferedReader(metaPath)) {
            String first = br.readLine();
            if(first==null || !first.startsWith("#recordLength="))
                throw new IOException("Invalid meta file: missing recordLength");
            int recLen = Integer.parseInt(first.substring("#recordLength=".length()));
            List<Scalar> cols = new ArrayList<>();
            List<Integer> fixedLens = new ArrayList<>();
            List<Integer> offs = new ArrayList<>();
            String line;
            int offset=0;
            while((line= br.readLine())!=null) {
                if(line.isBlank()) continue;
                String[] parts = line.split(",");
                if(parts.length<2) throw new IOException("Invalid meta line");
                String name = parts[0];
                IOMemento type = IOMemento.valueOf(parts[1]);
                int fixedLen = parts.length>=3? Integer.parseInt(parts[2]) : -1;
                cols.add(Scalar.of(type, name));
                fixedLens.add(fixedLen);
                offs.add(offset);
                offset += type==IOMemento.IO_STRING_FIXED? fixedLen : type.fixedSize();
            }
            if(offset!=recLen) throw new IOException("Record length mismatch");
            return new ISAMMeta(cols,fixedLens,offs,recLen);
        }
    }

    public void write(Path metaPath) throws IOException {
        try (BufferedWriter bw = Files.newBufferedWriter(metaPath)) {
            bw.write("#recordLength=" + recordLength);
            bw.newLine();
            for(int i=0;i<columns.size();i++) {
                Scalar s = columns.get(i);
                int fixedLen = fixedStringLengths.get(i);
                StringJoiner joiner = new StringJoiner(",");
                joiner.add(s.name()).add(s.type().name()).add(Integer.toString(fixedLen));
                bw.write(joiner.toString());
                bw.newLine();
            }
        }
    }

    public int columnCount() { return columns.size(); }

    public int offset(int column) { return offsets.get(column); }

    public int fixedStringLength(int column) { return fixedStringLengths.get(column); }
}