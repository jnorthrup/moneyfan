package com.moneyfan.io;

import com.moneyfan.core.IOMemento;
import com.moneyfan.core.Scalar;
import com.moneyfan.grid.Cell;
import com.moneyfan.grid.GridCursor;
import com.moneyfan.grid.RowVec;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Writes a GridCursor into ISAM binary file along with .meta sidecar.
 */
public final class ISAMWriter {

    private ISAMWriter() {}

    public static void write(GridCursor grid, Path dataPath) throws IOException {
        if(grid.rowCount()==0) throw new IllegalArgumentException("Grid is empty");
        List<Scalar> scalars = grid.getScalars();
        int colCount = scalars.size();

        // Determine string fixed lengths
        List<Integer> fixedLens = new ArrayList<>(colCount);
        for(int c=0; c<colCount; c++) {
            IOMemento type = scalars.get(c).type();
            if(type==IOMemento.IO_STRING_FIXED) {
                // compute max length across grid for this column
                int max = 0;
                for(int r=0;r<grid.rowCount();r++) {
                    Object val = grid.getRow(r).get(c).value();
                    int len = val==null?0: val.toString().length();
                    if(len>max) max=len;
                }
                fixedLens.add(max);
            } else {
                fixedLens.add(-1);
            }
        }

        ISAMMeta meta = ISAMMeta.fromColumns(scalars, fixedLens);
        Path metaPath = replaceExtension(dataPath, ".meta");
        meta.write(metaPath);

        try(FileChannel channel = FileChannel.open(dataPath, StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
            ByteBuffer buffer = ByteBuffer.allocate(meta.recordLength());
            for(int r=0;r<grid.rowCount();r++) {
                buffer.clear();
                RowVec row = grid.getRow(r);
                for(int c=0;c<colCount;c++) {
                    IOMemento type = scalars.get(c).type();
                    int offset = meta.offset(c);
                    Cell cell = row.get(c);
                    writeValue(buffer, offset, cell.value(), type, fixedLens.get(c));
                }
                buffer.position(meta.recordLength());
                buffer.flip();
                channel.write(buffer);
            }
        }
    }

    private static void writeValue(ByteBuffer buf, int offset, Object value, IOMemento type, int fixedLen) {
        switch(type) {
            case IO_INT -> buf.putInt(offset, (Integer) value);
            case IO_LONG -> buf.putLong(offset, (Long) value);
            case IO_DOUBLE -> buf.putDouble(offset, (Double) value);
            case IO_LOCAL_DATE -> buf.putInt(offset, (int) ((java.time.LocalDate) value).toEpochDay());
            case IO_INSTANT -> {
                java.time.Instant inst = (java.time.Instant) value;
                buf.putLong(offset, inst.getEpochSecond());
                buf.putInt(offset + 8, inst.getNano());
            }
            case IO_STRING_FIXED -> {
                String str = (String) value;
                byte[] bytes = str.getBytes(java.nio.charset.StandardCharsets.UTF_8);
                if(bytes.length>fixedLen) throw new IllegalArgumentException("String exceeds fixed length");
                buf.position(offset);
                buf.put(bytes);
                // pad with zeros
                for(int i=bytes.length;i<fixedLen;i++) buf.put((byte)0);
            }
        }
    }

    private static Path replaceExtension(Path path, String newExt) {
        String filename = path.getFileName().toString();
        int idx = filename.lastIndexOf('.');
        if(idx!=-1) filename = filename.substring(0, idx);
        return path.resolveSibling(filename + newExt);
    }
}