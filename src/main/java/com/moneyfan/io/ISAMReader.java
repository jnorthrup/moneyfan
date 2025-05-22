package com.moneyfan.io;

import com.moneyfan.core.CellMeta;
import com.moneyfan.core.Scalar;
import com.moneyfan.grid.Cell;
import com.moneyfan.grid.GridCursor;
import com.moneyfan.grid.RowVec;
import com.moneyfan.grid.Vect0r;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * Reads ISAM data files into a lazy, mmap-backed GridCursor.
 */
public class ISAMReader implements AutoCloseable {
    
    private final Path dataPath;
    private final ISAMMeta meta;
    private FileChannel fileChannel;
    private MappedByteBuffer mappedBuffer;
    private long fileSize;
    private int rowCount;
    
    public ISAMReader(Path dataPath, Path metaPath) throws IOException {
        this.dataPath = dataPath;
        this.meta = ISAMMeta.fromFile(metaPath);
    }
    
    public GridCursor open() throws IOException {
        fileChannel = FileChannel.open(dataPath, StandardOpenOption.READ);
        fileSize = fileChannel.size();
        rowCount = (int) (fileSize / meta.recordLength());
        
        // Map the entire file for simplicity
        mappedBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize);
        
        return createGridCursor();
    }
    
    private GridCursor createGridCursor() {
        Vect0r<RowVec> rows = Vect0r.of(rowCount, this::createRowVec);
        return GridCursor.of(rows);
    }
    
    private RowVec createRowVec(int rowIndex) {
        int baseOffset = rowIndex * meta.recordLength();
        Vect0r<Cell> cells = Vect0r.of(meta.getColumnCount(), 
            colIndex -> createCell(baseOffset, colIndex));
        return RowVec.of(cells);
    }
    
    private Cell createCell(int baseOffset, int colIndex) {
        Scalar scalar = meta.getScalarForColumn(colIndex);
        int offset = baseOffset + meta.getOffsetForColumn(colIndex);
        
        Object value;
        if (scalar.type() == com.moneyfan.core.IOMemento.IO_STRING_FIXED) {
            FixedDriver<String> driver = FixedDriver.stringDriver(scalar.stringLength());
            value = driver.read(mappedBuffer, offset);
        } else {
            FixedDriver<?> driver = FixedDriver.MAPPED_DRIVERS.get(scalar.type());
            value = driver.read(mappedBuffer, offset);
        }
        
        return Cell.of(value, CellMeta.of(scalar));
    }
    
    @Override
    public void close() throws IOException {
        if (fileChannel != null && fileChannel.isOpen()) {
            fileChannel.close();
        }
    }
}