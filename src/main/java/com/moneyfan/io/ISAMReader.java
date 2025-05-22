package com.moneyfan.io;

import com.moneyfan.core.CellMeta;
import com.moneyfan.core.Scalar;
import com.moneyfan.grid.Cell;
import com.moneyfan.grid.GridCursor;
import com.moneyfan.grid.RowVec;
import com.moneyfan.grid.Vect0r;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.function.Supplier;

/**
 * Opens an ISAM data file (.bin) along with its .meta sidecar and exposes a lazy GridCursor backed by mmap buffer.
 */
public final class ISAMReader implements AutoCloseable {

    private final Path dataPath;
    private final Path metaPath;
    private final FileChannel channel;
    private final ByteBuffer mapped;
    private final ISAMMeta meta;
    private final int rowCount;
    private final long windowSize;
    private long windowStart;
    private ByteBuffer windowBuf;

    public ISAMReader(Path dataPath) throws IOException {
        this(dataPath, 64 * 1024 * 1024); // 64MB default window
    }

    public ISAMReader(Path dataPath, long windowSize) throws IOException {
        this.dataPath = dataPath;
        this.metaPath = replaceExtension(dataPath, ".meta");
        this.meta = ISAMMeta.read(metaPath);
        this.channel = FileChannel.open(dataPath, StandardOpenOption.READ);
        long size = channel.size();
        if(size % meta.recordLength()!=0) throw new IOException("Data file size is not multiple of record length");
        this.rowCount = (int)(size / meta.recordLength());
        this.windowSize = windowSize;
        this.windowStart = 0;
        long mapSize = Math.min(size, windowSize);
        this.windowBuf = channel.map(FileChannel.MapMode.READ_ONLY, windowStart, mapSize);
        this.mapped = null; // legacy field unused now
    }

    public GridCursor open() {
        Vect0r<RowVec> rows = Vect0r.of(rowCount, this::rowAt);
        return new GridCursor(rows);
    }

    private RowVec rowAt(int index) {
        int recLen = meta.recordLength();
        long bytePos = (long) index * recLen;
        ensureWindow(bytePos, recLen);
        ByteBuffer dup = windowBuf.duplicate();
        int offsetInWindow = (int) (bytePos - windowStart);
        dup.position(offsetInWindow).limit(offsetInWindow + recLen);
        ByteBuffer slice = dup.slice();
        List<Scalar> scalars = meta.columns();
        Vect0r<Cell> cells = Vect0r.of(scalars.size(), col -> {
            Scalar sc = scalars.get(col);
            Object value = readValue(slice, meta.offset(col), sc);
            Supplier<Scalar> sup = () -> sc;
            return new Cell(value, new CellMeta(sup));
        });
        return new RowVec(cells);
    }

    private Object readValue(ByteBuffer buf, int offset, Scalar scalar) {
        return switch (scalar.type()) {
            case IO_INT -> buf.getInt(offset);
            case IO_LONG -> buf.getLong(offset);
            case IO_DOUBLE -> buf.getDouble(offset);
            case IO_LOCAL_DATE -> java.time.LocalDate.ofEpochDay(buf.getInt(offset));
            case IO_INSTANT -> {
                long sec = buf.getLong(offset);
                int nanos = buf.getInt(offset+8);
                yield java.time.Instant.ofEpochSecond(sec, nanos);
            }
            case IO_STRING_FIXED -> {
                int len = meta.fixedStringLength(meta.columns().indexOf(scalar));
                byte[] bytes = new byte[len];
                buf.position(offset);
                buf.get(bytes);
                // remove zero padding
                int actualLen=len;
                for(int i=0;i<len;i++) if(bytes[i]==0) { actualLen=i; break; }
                yield new String(bytes, 0, actualLen, java.nio.charset.StandardCharsets.UTF_8);
            }
        };
    }

    private static Path replaceExtension(Path path, String newExt) {
        String filename = path.getFileName().toString();
        int idx = filename.lastIndexOf('.');
        if(idx!=-1) filename = filename.substring(0, idx);
        return path.resolveSibling(filename + newExt);
    }

    private void ensureWindow(long bytePos, int length) {
        if(bytePos < windowStart || bytePos + length > windowStart + windowBuf.capacity()) {
            // remap window
            windowStart = (bytePos / windowSize) * windowSize;
            long remaining = 0;
            try {
                remaining = channel.size() - windowStart;
                long mapSize = Math.min(windowSize, remaining);
                windowBuf = channel.map(FileChannel.MapMode.READ_ONLY, windowStart, mapSize);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    @Override
    public void close() throws IOException {
        channel.close();
    }
}