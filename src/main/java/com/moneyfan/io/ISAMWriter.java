package com.moneyfan.io;

import com.moneyfan.core.IOMemento;
import com.moneyfan.core.Scalar;
import com.moneyfan.grid.Cell;
import com.moneyfan.grid.GridCursor;
import com.moneyfan.grid.RowVec;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;

/**
 * Writes a GridCursor to ISAM format.
 */
public class ISAMWriter implements AutoCloseable {
    
    private final Path dataPath;
    private final Path metaPath;
    private final GridCursor cursor;
    private final ISAMMeta meta;
    private FileChannel fileChannel;
    
    public ISAMWriter(GridCursor cursor, Path dataPath, Path metaPath) {
        this.cursor = cursor;
        this.dataPath = dataPath;
        this.metaPath = metaPath;
        this.meta = ISAMMeta.fromScalars(cursor.getScalars());
    }
    
    public void write() throws IOException {
        // Write metadata
        meta.writeToFile(metaPath);
        
        // Open file channel for writing data
        fileChannel = FileChannel.open(dataPath, 
            StandardOpenOption.WRITE, 
            StandardOpenOption.CREATE, 
            StandardOpenOption.TRUNCATE_EXISTING);
        
        ByteBuffer buffer = ByteBuffer.allocate(meta.recordLength());
        
        // Write each row
        for (int rowIndex = 0; rowIndex < cursor.rowCount(); rowIndex++) {
            writeRow(cursor.getRow(rowIndex), buffer);
            
            // Reset buffer for writing
            buffer.flip();
            
            // Write buffer to file
            fileChannel.write(buffer);
            
            // Clear buffer for next row
            buffer.clear();
        }
    }
    
    private void writeRow(RowVec row, ByteBuffer buffer) {
        List<Scalar> scalars = meta.scalars();
        List<Integer> offsets = meta.offsets();
        
        for (int colIndex = 0; colIndex < scalars.size(); colIndex++) {
            Scalar scalar = scalars.get(colIndex);
            int offset = offsets.get(colIndex);
            Cell cell = row.getCell(colIndex);
            Object value = cell.value();
            
            if (value == null) {
                // Write default values for null
                continue;
            }
            
            if (scalar.type() == IOMemento.IO_STRING_FIXED) {
                FixedDriver<String> driver = FixedDriver.stringDriver(scalar.stringLength());
                driver.write(buffer, offset, (String) value);
            } else {
                FixedDriver<?> driver = FixedDriver.MAPPED_DRIVERS.get(scalar.type());
                writeValue(driver, buffer, offset, value);
            }
        }
    }
    
    @SuppressWarnings("unchecked")
    private <T> void writeValue(FixedDriver<T> driver, ByteBuffer buffer, int offset, Object value) {
        driver.write(buffer, offset, (T) value);
    }
    
    @Override
    public void close() throws IOException {
        if (fileChannel != null && fileChannel.isOpen()) {
            fileChannel.close();
        }
    }
    
    public static void writeGridCursor(GridCursor cursor, Path dataPath, Path metaPath) throws IOException {
        try (ISAMWriter writer = new ISAMWriter(cursor, dataPath, metaPath)) {
            writer.write();
        }
    }
}