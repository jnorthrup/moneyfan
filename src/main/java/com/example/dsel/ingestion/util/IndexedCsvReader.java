package com.example.dsel.ingestion.util;

import java.io.BufferedReader;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Iterator;

public class IndexedCsvReader implements Closeable, Iterable<String> {

    private final Path csvFilePath;
    private final FileChannel fileChannel;
    private final List<Long> lineOffsets = new ArrayList<>();
    private String headerLine;

    public IndexedCsvReader(Path csvFilePath) throws IOException {
        this.csvFilePath = csvFilePath;
        this.fileChannel = FileChannel.open(csvFilePath, StandardOpenOption.READ);
        buildIndex();
    }

    private void buildIndex() throws IOException {
        // Use BufferedReader to correctly handle line endings and character encoding.
        // The channel's position will be advanced by the reader.
        // We need to track raw byte offsets, so we use a custom approach.
        
        fileChannel.position(0); // Start from the beginning
        InputStreamReader isr = new InputStreamReader(Channels.newInputStream(fileChannel), StandardCharsets.UTF_8);
        // Not using BufferedReader.readLine() for indexing to get precise byte offsets.
        // Instead, read char by char or byte by byte to find EOLs.

        StringBuilder currentLineContent = new StringBuilder();
        long currentByteOffset = 0;
        int bytesRead;
        boolean isFirstLine = true;
        
        // Reading byte by byte to precisely track offsets before each line.
        // This is more robust than using readLine() and guessing EOL lengths.
        ByteBuffer buffer = ByteBuffer.allocate(1); // Read one byte at a time
        
        // Header line
        lineOffsets.add(currentByteOffset); // Offset of the header line itself is 0

        while (fileChannel.read(buffer) != -1) {
            buffer.flip();
            byte b = buffer.get();
            buffer.clear();
            currentByteOffset++;

            if (b == '\n') { // LF
                if (isFirstLine) {
                    headerLine = currentLineContent.toString();
                    isFirstLine = false;
                }
                // Add offset for the *start* of the next line
                if (fileChannel.position() < fileChannel.size()) { // Avoid adding offset if it's the last EOL
                     lineOffsets.add(currentByteOffset);
                }
                currentLineContent.setLength(0);
            } else if (b == '\r') { // CR
                // Could be CR or CRLF. If next is LF, the LF handler will take care of it.
                // If it's just CR (old Mac), this is the EOL.
                if (isFirstLine) {
                    headerLine = currentLineContent.toString();
                    isFirstLine = false;
                }
                // Check if next is LF
                if (fileChannel.position() < fileChannel.size()) {
                    ByteBuffer lookAheadBuffer = ByteBuffer.allocate(1);
                    fileChannel.read(lookAheadBuffer);
                    lookAheadBuffer.flip();
                    byte nextByte = lookAheadBuffer.get();
                    lookAheadBuffer.clear();
                    
                    if (nextByte == '\n') { // CRLF
                        currentByteOffset++; // Account for LF
                    } else {
                        // Not LF, so it was a lone CR. Rewind channel for the nextByte.
                        fileChannel.position(fileChannel.position() - 1);
                    }
                }
                if (fileChannel.position() < fileChannel.size()) {
                    lineOffsets.add(currentByteOffset);
                }
                currentLineContent.setLength(0);
            } else {
                currentLineContent.append((char) b);
            }
        }
        
        // In case the file doesn't end with a newline
        if (isFirstLine && currentLineContent.length() > 0) { // File with only one line, no EOL
            headerLine = currentLineContent.toString();
        } else if (!isFirstLine && currentLineContent.length() > 0) {
            // This case (last line has content but no EOL) means its offset was added,
            // but it wasn't processed into headerLine if it was the only line.
            // The logic for lineOffsets.add() above correctly captures the start of this last line.
        }


        if (lineOffsets.isEmpty() && headerLine != null) { // Only header, no data lines
             // No data lines to add to lineOffsets beyond the header's implicit 0
        } else if (!lineOffsets.isEmpty()) {
             // The first offset (0) is for the header. Remove it from data line offsets.
             // The header itself is stored separately.
             lineOffsets.remove(0);
        }


        fileChannel.position(0); // Reset for subsequent reads by getDataLine
    }
    
    public String getHeader() {
        return headerLine;
    }

    public int getNumberOfDataLines() {
        return lineOffsets.size();
    }

    public String getDataLine(int lineNumber) throws IOException {
        if (lineNumber < 0 || lineNumber >= lineOffsets.size()) {
            throw new IndexOutOfBoundsException("Line number out of bounds: " + lineNumber + " (total data lines: " + lineOffsets.size() + ")");
        }
        long offset = lineOffsets.get(lineNumber);
        
        fileChannel.position(offset);
        // Use BufferedReader to read a single line from this specific offset.
        // This is fine as BufferedReader will read until EOL from that point.
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(Channels.newInputStream(fileChannel), StandardCharsets.UTF_8))) {
            String line = reader.readLine();
            if (line == null) { 
                throw new IOException("Failed to read line at offset " + offset + " for line number " + lineNumber);
            }
            return line;
        }
    }

    @Override
    public Iterator<String> iterator() {
        return new Iterator<String>() {
            private int currentLine = 0;

            @Override
            public boolean hasNext() {
                return currentLine < getNumberOfDataLines();
            }

            @Override
            public String next() {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }
                try {
                    return getDataLine(currentLine++);
                } catch (IOException e) {
                    throw new RuntimeException("Error reading line in iterator", e);
                }
            }
        };
    }

    @Override
    public void close() throws IOException {
        if (fileChannel != null && fileChannel.isOpen()) {
            fileChannel.close();
        }
    }
}
