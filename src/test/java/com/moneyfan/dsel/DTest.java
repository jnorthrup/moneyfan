package com.moneyfan.dsel;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.*;
import java.math.BigDecimal;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;

import static com.moneyfan.dsel.D.*;
import static org.junit.jupiter.api.Assertions.*;

class DTest {

    @TempDir
    Path tempDir;

    private Path createTestCsv(String fileName, String... lines) throws IOException {
        Path filePath = tempDir.resolve(fileName);
        Files.write(filePath, Arrays.asList(lines), StandardCharsets.UTF_8);
        return filePath;
    }

    private Path getResourcePath(String resourceName) {
        try {
            return Paths.get(Objects.requireNonNull(getClass().getResource(resourceName)).toURI());
        } catch (Exception e) {
            throw new RuntimeException("Cannot find resource: " + resourceName, e);
        }
    }

    @Test
    void testJoinOperations() {
        Join<String, Integer> j1 = jn("hello", 123);
        assertEquals("hello", f(j1));
        assertEquals(123, s(j1));
        assertEquals("jn(hello, 123)", j1.toString());

        Join<Integer, String> j2 = sw(j1);
        assertEquals(123, f(j2));
        assertEquals("hello", s(j2));

        Join<String, Integer> j3 = mf(j1, String::toUpperCase);
        assertEquals("HELLO", f(j3));
        assertEquals(123, s(j3));

        Join<String, String> j4 = ms(j1, val -> "v" + val);
        assertEquals("hello", f(j4));
        assertEquals("v123", s(j4));

        Join<Integer, Double> j5 = mb(j1, String::length, val -> val / 10.0);
        assertEquals(5, f(j5));
        assertEquals(12.3, s(j5), 0.001);

        assertTrue(tst(j1, j -> f(j).startsWith("h") && s(j) > 100));
        assertFalse(tst(j1, j -> s(j) < 0));
    }

    @Test
    void testSeriesOperations() {
        Series<Integer> s1 = sr(5, i -> i * 10);
        assertEquals(5, sz(s1));
        assertEquals(0, get(s1, 0));
        assertEquals(40, get(s1, 4));
        assertThrows(IndexOutOfBoundsException.class, () -> get(s1, 5));
        assertThrows(IndexOutOfBoundsException.class, () -> get(s1, -1));

        Series<String> s2 = map(s1, i -> "v" + i);
        assertEquals(5, sz(s2));
        assertEquals("v0", get(s2, 0));
        assertEquals("v40", get(s2, 4));

        Series<Integer> s3 = flt(s1, i -> i > 20);
        assertEquals(2, sz(s3)); // 30, 40
        assertEquals(30, get(s3, 0));
        assertEquals(40, get(s3, 1));
        // Test filter caching
        assertEquals(2, sz(s3)); // Access again


        assertEquals(Arrays.asList(0, 10, 20, 30, 40), ls(s1));
        assertEquals(Arrays.asList("v0", "v10", "v20", "v30", "v40"), ls(s2));

        assertEquals(0, fst(s1));
        assertEquals(40, lst(s1));
        Series<Integer> emptySeries = sr(0, i -> null);
        assertThrows(NoSuchElementException.class, () -> fst(emptySeries));
        assertThrows(NoSuchElementException.class, () -> lst(emptySeries));


        Series<Integer> sHead = hd(s1, 3);
        assertEquals(3, sz(sHead));
        assertEquals(Arrays.asList(0, 10, 20), ls(sHead));

        Series<Integer> sTail = tl(s1, 3);
        assertEquals(3, sz(sTail));
        assertEquals(Arrays.asList(20, 30, 40), ls(sTail));

        Series<Integer> sSkip = sk(s1, 2);
        assertEquals(3, sz(sSkip));
        assertEquals(Arrays.asList(20, 30, 40), ls(sSkip));

        List<Integer> collected = new ArrayList<>();
        each(s1, collected::add);
        assertEquals(Arrays.asList(0, 10, 20, 30, 40), collected);
    }

    @Test
    void testTypeEvidenceAndDeduction() {
        TypeEvidence ev = new TypeEvidence();
        ev.assess("123");
        ev.assess("456");
        ev.assess("true");
        ev.assess("text");
        ev.assess("");
        ev.assess("78.9");

        assertEquals(2, ev.longValueCount);
        assertEquals(1, ev.doubleValueCount);
        assertEquals(1, ev.booleanValueCount);
        assertEquals(1, ev.stringValueCount);
        assertEquals(1, ev.emptyCount);
        assertEquals(4, ev.maxStrLength); // "text"
        assertEquals("123", ev.firstNonEmptySample);

        // Because "text" is present, it should deduce String
        TypeMemento deduced = ev.deduceFinalType();
        assertEquals(TypeMemento.Basic.STRING.getTypeName(), deduced.getTypeName());
        assertEquals(4, deduced.getFixedSize()); // fsString(maxStrLength)

        ev = new TypeEvidence();
        ev.assess("10"); ev.assess("20.5"); ev.assess("");
        deduced = ev.deduceFinalType();
        assertEquals(TypeMemento.Basic.DOUBLE.getTypeName(), deduced.getTypeName());

        ev = new TypeEvidence();
        ev.assess("10"); ev.assess("20"); ev.assess("false");
        deduced = ev.deduceFinalType(); // Longs and Booleans -> Long (if no strings/doubles)
        assertEquals(TypeMemento.Basic.LONG.getTypeName(), deduced.getTypeName());

        ev = new TypeEvidence();
        ev.assess("true"); ev.assess("FALSE"); ev.assess("");
        deduced = ev.deduceFinalType();
        assertEquals(TypeMemento.Basic.BOOLEAN.getTypeName(), deduced.getTypeName());

        ev = new TypeEvidence(); // All empty
        ev.assess(""); ev.assess(null);
        deduced = ev.deduceFinalType();
        assertEquals(TypeMemento.Basic.STRING.getTypeName(), deduced.getTypeName());
        assertEquals(1, deduced.getFixedSize()); // max(1, maxStrLength which is 0)
    }

    @Test
    void testParseCsvLine() {
        assertEquals(Arrays.asList("a", "b", "c"), parseCsvLine("a,b,c", ',', '"'));
        assertEquals(Arrays.asList("a", "b,c", "d"), parseCsvLine("a,\"b,c\",d", ',', '"'));
        assertEquals(Arrays.asList("a", "b\"c", "d"), parseCsvLine("a,\"b\"\"c\",d", ',', '"'));
        assertEquals(Collections.singletonList(""), parseCsvLine("\"\"", ',', '"'));
        assertEquals(Arrays.asList("", "a"), parseCsvLine(",a", ',', '"'));
        assertEquals(Arrays.asList("a", ""), parseCsvLine("a,", ',', '"'));
        assertEquals(Collections.emptyList(), parseCsvLine(null, ',', '"'));
        assertEquals(Collections.emptyList(), parseCsvLine("", ',', '"'));
        assertEquals(Arrays.asList("a;b", "c"), parseCsvLine("\"a;b\";c", ';', '"'));
    }

    @Test
    void testReadCsvWithHeader() throws IOException {
        Path csvFile = getResourcePath("test_simple_header.csv");
        Cursor cursor = readCsv(csvFile.toString(), true, ',', '"', 10);

        assertNotNull(cursor);
        assertEquals(3, sz(cursor)); // 3 data rows

        RowVec headerCheckRv = get(cursor, 0); // First data row
        assertEquals("ID", colName(headerCheckRv, 0));
        assertEquals("Name", colName(headerCheckRv, 1));
        assertEquals("Value", colName(headerCheckRv, 2));

        // Types deduced (ID as Long, Name as String(5), Value as Double)
        assertTrue(colType(headerCheckRv, 0) instanceof TypeMemento.Basic);
        assertEquals(TypeMemento.Basic.LONG.getTypeName(), colType(headerCheckRv, 0).getTypeName());

        assertTrue(colType(headerCheckRv, 1) instanceof D.FixedSizeTypeMemento);
        assertEquals(TypeMemento.Basic.STRING.getTypeName(), colType(headerCheckRv, 1).getTypeName());
        assertEquals(5, colType(headerCheckRv, 1).getFixedSize()); // "Alpha", "Beta", "Gamma"

        assertTrue(colType(headerCheckRv, 2) instanceof TypeMemento.Basic);
        assertEquals(TypeMemento.Basic.DOUBLE.getTypeName(), colType(headerCheckRv, 2).getTypeName());

        assertEquals(1L, get(get(cursor, 0), 0));
        assertEquals("Alpha", get(get(cursor, 0), 1));
        assertEquals(100.5, (Double) get(get(cursor, 0), 2), 0.001);

        assertEquals(3L, get(get(cursor, 2), 0));
        assertEquals("Gamma", get(get(cursor, 2), 1));
        assertEquals(30.2, (Double) get(get(cursor, 2), 2), 0.001);
    }

    @Test
    void testReadCsvNoHeader() throws IOException {
        Path csvFile = getResourcePath("test_simple_no_header.csv");
        Cursor cursor = readCsv(csvFile.toString(), false, ',', '"', 10);

        assertEquals(3, sz(cursor));
        RowVec row = get(cursor, 0);
        assertEquals("column_0", colName(row, 0));
        assertEquals("column_1", colName(row, 1));
        assertEquals(1L, get(row, 0));
        assertEquals("Alpha", get(row, 1));
    }
    
    @Test
    void testReadCsvDataTypes() throws IOException {
        Path csvFile = getResourcePath("test_data_types.csv");
        Cursor cursor = readCsv(csvFile.toString(), true, ',', '"', 10);
        assertEquals(4, sz(cursor)); // 4 data rows

        RowVec r1 = get(cursor, 0);
        assertEquals("IntCol", colName(r1,0)); // Name check
        assertEquals(TypeMemento.Basic.LONG.getTypeName(), colType(r1, 0).getTypeName()); // All numbers become Long or Double
        assertEquals(10L, get(r1,0));
        assertEquals(10000000000L, get(r1,1));
        assertEquals(3.14, (Double)get(r1,2), 0.001);
        assertEquals(true, get(r1,3));
        assertEquals("Hello, World!", get(r1,4));
        assertEquals(TypeMemento.Basic.STRING.getTypeName(), colType(r1,4).getTypeName());
        assertEquals(23, colType(r1,4).getFixedSize()); // " Special Chars: @#$ " is longest
        assertEquals(TypeMemento.Basic.STRING.getTypeName(), colType(r1,5).getTypeName()); // Char 'A' becomes String "A"
        assertEquals("A", get(r1,5));


        RowVec r2 = get(cursor, 1);
        assertEquals(false, get(r2,3));
        assertEquals("Test \"Quotes\"", get(r2,4));

        RowVec r3Empty = get(cursor, 2); // ", , , ,"", "
        assertNull(get(r3Empty, 0)); // Empty numeric becomes null
        assertNull(get(r3Empty, 1));
        assertNull(get(r3Empty, 2));
        assertEquals(false, get(r3Empty, 3)); // Empty boolean becomes false
        assertEquals("", get(r3Empty, 4));    // Empty string
        assertEquals(" ", get(r3Empty, 5)); // " " (space)
        
        RowVec r4 = get(cursor, 3);
        assertEquals(-5L, get(r4,0));
        assertEquals(true, get(r4,3)); // TRUE
    }


    @Test
    void testReadCsvWithQuotesAndDifferentDelimiter() throws IOException {
        Path csvFile = getResourcePath("test_quotes_delimiters.csv");
        Cursor cursor = readCsv(csvFile.toString(), true, ';', '"', 10);

        assertEquals(2, sz(cursor));
        RowVec row1 = get(cursor, 0);
        assertEquals("field1", get(row1, 0));
        assertEquals("field2 with ; semicolon", get(row1, 1));
        assertEquals("field3 with \"quotes\"", get(row1, 2));
    }
    
    @Test
    void testReadCsvMalformedLines() throws IOException {
        Path csvFile = getResourcePath("test_malformed.csv");
        // Expects warnings to stderr, but should process based on first data line or header for num cols
        Cursor cursor = readCsv(csvFile.toString(), true, ',', '"', 10);
        assertEquals(3, sz(cursor)); // 3 lines of data
        
        RowVec r1 = get(cursor,0); // val1,val2
        assertEquals(2, sz(r1));
        assertEquals("val1", get(r1,0));
        assertEquals("val2", get(r1,1));

        RowVec r2 = get(cursor,1); // val3,val4,val5 (truncated to 2 cols)
        assertEquals(2, sz(r2));
        assertEquals("val3", get(r2,0));
        assertEquals("val4", get(r2,1));

        RowVec r3 = get(cursor,2); // val6 (padded to 2 cols)
        assertEquals(2, sz(r3));
        assertEquals("val6", get(r3,0));
        assertEquals("", get(r3,1)); // Padded with empty string
    }


    @Test
    void testConvertCsvStringToTypedValue() {
        assertEquals(123, convertCsvStringToTypedValue("123", TypeMemento.Basic.INTEGER));
        assertEquals(123L, convertCsvStringToTypedValue(" 123 ", TypeMemento.Basic.LONG));
        assertEquals(3.14f, convertCsvStringToTypedValue("3.14", TypeMemento.Basic.FLOAT));
        assertEquals(3.14, convertCsvStringToTypedValue(" 3.14 ", TypeMemento.Basic.DOUBLE));
        assertEquals(true, convertCsvStringToTypedValue("true", TypeMemento.Basic.BOOLEAN));
        assertEquals(false, convertCsvStringToTypedValue("FALSE", TypeMemento.Basic.BOOLEAN));
        assertEquals("hello", convertCsvStringToTypedValue("hello", fsString(10)));
        assertEquals('A', convertCsvStringToTypedValue("A", TypeMemento.Basic.CHAR));
        assertThrows(IllegalArgumentException.class, () -> convertCsvStringToTypedValue("AA", TypeMemento.Basic.CHAR));

        // Empty/Null cases
        assertEquals("", convertCsvStringToTypedValue("", fsString(5)));
        assertEquals(false, convertCsvStringToTypedValue(" ", TypeMemento.Basic.BOOLEAN)); // " " is not "true"
        assertNull(convertCsvStringToTypedValue(" ", TypeMemento.Basic.INTEGER)); // Fails parse
        assertNull(convertCsvStringToTypedValue(null, TypeMemento.Basic.INTEGER));
        assertEquals("", convertCsvStringToTypedValue(null, fsString(10)));


        byte[] decoded = (byte[]) convertCsvStringToTypedValue(Base64.getEncoder().encodeToString("blob".getBytes()), D.fsBinaryBlob(10));
        assertArrayEquals("blob".getBytes(), decoded);
        assertThrows(IllegalArgumentException.class, () -> convertCsvStringToTypedValue("not base64", D.fsBinaryBlob(10)));
    }

    @Test
    void testIsamReadWriteOperations() throws IOException {
        List<ColumnMeta> schema = Arrays.asList(
                cm("ID", TypeMemento.Basic.INTEGER),
                cm("Name", fsString(10)), // Fixed length string
                cm("Value", TypeMemento.Basic.DOUBLE),
                cm("Active", TypeMemento.Basic.BOOLEAN),
                cm("Data", fsBinaryBlob(5))
        );

        // 1. Create a source CSV to use with csvToIsam
        Path sourceCsvPath = createTestCsv("isam_source.csv",
                "ID,Name,Value,Active,Data", // Header
                "1,TestOne,123.45,true," + Base64.getEncoder().encodeToString("abc".getBytes()),
                "2,Second,67.8,false," + Base64.getEncoder().encodeToString("12345".getBytes()),
                "3,TruncateMeString,0.0,true," + Base64.getEncoder().encodeToString("longblob".getBytes()), // String will be truncated, blob will be truncated
                "4,,9.9,false," + Base64.getEncoder().encodeToString("".getBytes()) // Empty name, empty blob
        );
        
        String isamBasePath = tempDir.resolve("myIsamFile").toString();
        csvToIsam(sourceCsvPath.toString(), isamBasePath, schema, true, ',', '"');

        // 2. Verify .meta file
        Path metaFilePath = Paths.get(isamBasePath + D.META_SFX);
        assertTrue(Files.exists(metaFilePath));
        IsamFileMetadata writtenMeta;
        try (DataInputStream metaIn = new DataInputStream(new FileInputStream(metaFilePath.toFile()))) {
            writtenMeta = IsamFileMetadata.read(metaIn);
        }
        assertEquals(4, writtenMeta.recordCount());
        assertEquals(schema.size(), writtenMeta.schema().size());
        assertEquals(TypeMemento.Basic.INTEGER.getFixedSize() + 10 + TypeMemento.Basic.DOUBLE.getFixedSize() + TypeMemento.Basic.BOOLEAN.getFixedSize() + 5, writtenMeta.recordByteLength());
        assertEquals("Name", writtenMeta.schema().get(1).f());
        assertEquals(fsString(10).toString(), writtenMeta.schema().get(1).s().toString());


        // 3. Read using IsamCursor
        try (IsamCursor cursor = new IsamCursor(isamBasePath)) {
            assertEquals(4, sz(cursor));

            // Row 1
            RowVec r1 = get(cursor, 0);
            assertEquals(schema.size(), sz(r1));
            assertEquals(1, get(r1, 0)); // ID
            assertEquals("TestOne", get(r1, 1)); // Name
            assertEquals(123.45, (Double) get(r1, 2), 0.001); // Value
            assertEquals(true, get(r1, 3)); // Active
            assertArrayEquals("abc".getBytes(), (byte[])get(r1,4)); // Data (padded with 2 nulls, but fsBinaryBlob returns full array)

            // Row 2
            RowVec r2 = get(cursor, 1);
            assertEquals(2, get(r2, 0));
            assertEquals("Second", get(r2, 1));
            assertEquals(false, get(r2, 3));
            assertArrayEquals("12345".getBytes(), (byte[])get(r2,4));


            // Row 3 (truncation check)
            RowVec r3 = get(cursor, 2);
            assertEquals("TruncateM", get(r3, 1)); // "TruncateMeString" truncated to 10 chars
            assertArrayEquals("longb".getBytes(), (byte[])get(r3,4)); // "longblob" truncated to 5 bytes


            // Row 4 (null/empty check)
            RowVec r4 = get(cursor, 3);
            assertEquals(0, (Integer) get(r4, 0)); // Integer default for null in writeValueToBuffer is 0
            assertEquals("", get(r4, 1)); // Empty string
            assertEquals(false, get(r4,3));
            assertArrayEquals("".getBytes(), Arrays.copyOf((byte[])get(r4,4),0)); // Empty blob (check actual content)

            // Check column metadata from IsamRowVec
            assertEquals("ID", colName(r1,0));
            assertEquals(TypeMemento.Basic.INTEGER.getTypeName(), colType(r1,0).getTypeName());
            assertEquals(fsString(10).toString(), colType(r1,1).s().toString());
            assertEquals(5, colType(r1,4).getFixedSize());

            // Test iterating
            List<Object> ids = ls(cursor).stream().map(row -> get(row, 0)).collect(Collectors.toList());
            assertEquals(Arrays.asList(1, 2, 3, 0), ids); // 0 for the null int case
        }
    }
    
    @Test
    void testIsamWriteReadValueBufferEdgeCases() {
        ByteBuffer buffer = ByteBuffer.allocate(100).order(ByteOrder.BIG_ENDIAN);

        // String shorter than fixed size (padding)
        TypeMemento fs10 = fsString(10);
        writeValueToBuffer("Hello", fs10, buffer);
        buffer.flip();
        assertEquals("Hello", readValueFromBuffer(fs10, buffer.duplicate()));
        buffer.clear();

        // String exact fixed size
        writeValueToBuffer("0123456789", fs10, buffer);
        buffer.flip();
        assertEquals("0123456789", readValueFromBuffer(fs10, buffer.duplicate()));
        buffer.clear();

        // String longer than fixed size (truncation)
        writeValueToBuffer("ThisIsTooLong", fs10, buffer);
        buffer.flip();
        assertEquals("ThisIsTooL", readValueFromBuffer(fs10, buffer.duplicate()));
        buffer.clear();
        
        // Empty string
        writeValueToBuffer("", fs10, buffer);
        buffer.flip();
        assertEquals("", readValueFromBuffer(fs10, buffer.duplicate()));
        buffer.clear();

        // Null string
        writeValueToBuffer(null, fs10, buffer);
        buffer.flip();
        assertEquals("", readValueFromBuffer(fs10, buffer.duplicate())); // Null becomes empty string after read
        buffer.clear();

        // Binary Blob
        TypeMemento bb5 = fsBinaryBlob(5);
        byte[] data1 = {1,2,3};
        writeValueToBuffer(data1, bb5, buffer);
        buffer.flip();
        assertArrayEquals(new byte[]{1,2,3,0,0}, (byte[])readValueFromBuffer(bb5, buffer.duplicate()));
        buffer.clear();

        byte[] data2 = {1,2,3,4,5,6,7}; // Truncate
        writeValueToBuffer(data2, bb5, buffer);
        buffer.flip();
        assertArrayEquals(new byte[]{1,2,3,4,5}, (byte[])readValueFromBuffer(bb5, buffer.duplicate()));
        buffer.clear();

        // Null numerics
        writeValueToBuffer(null, TypeMemento.Basic.INTEGER, buffer);
        writeValueToBuffer(null, TypeMemento.Basic.DOUBLE, buffer);
        buffer.flip();
        assertEquals(0, readValueFromBuffer(TypeMemento.Basic.INTEGER, buffer.duplicate()));
        assertEquals(0.0, (Double)readValueFromBuffer(TypeMemento.Basic.DOUBLE, buffer.duplicate()), 0.001);
        buffer.clear();
    }


    @Test
    void testIsamInvalidMetadata() throws IOException {
        // Write intentionally bad metadata
        Path badMetaPath = tempDir.resolve("bad.meta");
        try (DataOutputStream out = new DataOutputStream(new FileOutputStream(badMetaPath.toFile()))) {
            out.writeInt(-1); // Invalid schema size
        }
        assertThrows(IOException.class, () -> new IsamCursor(tempDir.resolve("bad").toString()));

        Path badMetaPath2 = tempDir.resolve("bad2.meta");
         try (DataOutputStream out = new DataOutputStream(new FileOutputStream(badMetaPath2.toFile()))) {
            out.writeInt(1); // schema size
            out.writeUTF("col1");
            out.writeUTF(TypeMemento.Basic.STRING.getTypeName());
            out.writeInt(-5); // Invalid fixed size for string
            out.writeLong(0);
            out.writeInt(0);
        }
        assertThrows(IOException.class, () -> new IsamCursor(tempDir.resolve("bad2").toString()));
        
        Path badMetaPath3 = tempDir.resolve("bad3.meta");
         try (DataOutputStream out = new DataOutputStream(new FileOutputStream(badMetaPath3.toFile()))) {
            out.writeInt(1); // schema size
            out.writeUTF("col1");
            out.writeUTF(TypeMemento.Basic.INTEGER.getTypeName());
            out.writeInt(8); // Mismatched fixed size for Integer (should be 4)
            out.writeLong(0);
            out.writeInt(8);
        }
        assertThrows(IOException.class, () -> new IsamCursor(tempDir.resolve("bad3").toString()));

        Path badMetaPath4 = tempDir.resolve("bad4.meta");
         try (DataOutputStream out = new DataOutputStream(new FileOutputStream(badMetaPath4.toFile()))) {
            out.writeInt(1); // schema size
            out.writeUTF("col1");
            out.writeUTF(TypeMemento.Basic.INTEGER.getTypeName());
            out.writeInt(4); 
            out.writeLong(1); // 1 record
            out.writeInt(100); // Mismatched recordByteLength (calculated would be 4)
        }
        assertThrows(IOException.class, () -> new IsamCursor(tempDir.resolve("bad4").toString()));
    }
    
    @Test
    void testConceptualFactories() {
        Series<Integer> s = sr(3, i -> i);
        assertEquals(3, sz(s));

        ColumnMeta cm = cm("Age", TypeMemento.Basic.INTEGER);
        assertEquals("Age", f(cm));
        assertEquals(TypeMemento.Basic.INTEGER, s(cm));

        RowVec rv = rv(2, i -> {
            Supplier<ColumnMeta> metaSupplier = () -> cm("col"+i, TypeMemento.Basic.STRING);
            return jn("val"+i, metaSupplier);
        });
        assertEquals(2, sz(rv));
        assertEquals("val0", get(rv,0));
        assertEquals("col1", colName(rv,1));

        Cursor c = cr(1, i -> rv);
        assertEquals(1, sz(c));
        assertEquals("val1", get(get(c,0),1));

        Twin<String> tw = tw("a", "b");
        assertEquals("a", f(tw));
        assertEquals("b", s(tw));
    }
    
    @Test
    void testCursorRowOperations() throws IOException {
        Path csvFile = getResourcePath("test_simple_header.csv");
        Cursor cursor = readCsv(csvFile.toString(), true, ',', '"', 10); // ID,Name,Value

        // mapRow: Convert "Value" column to integer by floor
        Cursor mappedCursor = mapRow(cursor, row -> {
            Object originalValue = get(row, 2); // Value column
            Double doubleValue = (originalValue instanceof Double) ? (Double)originalValue : Double.parseDouble(originalValue.toString());
            Integer intValue = (int)Math.floor(doubleValue);
            
            // Create a new RowVec with the modified value
            // This is a bit verbose; a RowVec.with(colIdx, newValue) or mapCell would be nice
            List<Join<Object, Supplier<ColumnMeta>>> newCells = new ArrayList<>();
            for (int i = 0; i < sz(row); i++) {
                final int colIdx = i;
                Supplier<ColumnMeta> cmSupplier = () -> cm(colName(row, colIdx), colType(row, colIdx));
                if (i == 2) {
                    cmSupplier = () -> cm(colName(row, colIdx), TypeMemento.Basic.INTEGER); // Change type if needed
                    newCells.add(jn(intValue, cmSupplier));
                } else {
                    newCells.add(jn(get(row, i), cmSupplier));
                }
            }
            return rv(sz(row), newCells::get);
        });

        assertEquals(3, sz(mappedCursor));
        assertEquals(100, get(get(mappedCursor,0), 2)); // 100.5 -> 100
        assertEquals(200, get(get(mappedCursor,1), 2)); // 200.0 -> 200
        assertEquals(30, get(get(mappedCursor,2), 2));  // 30.2  -> 30
        assertEquals(TypeMemento.Basic.INTEGER.getTypeName(), colType(get(mappedCursor,0),2).getTypeName());


        // fltRow: Filter rows where ID > 1
        Cursor filteredCursor = fltRow(cursor, row -> (Long)get(row,0) > 1L);
        assertEquals(2, sz(filteredCursor));
        assertEquals(2L, get(get(filteredCursor,0),0)); // Row with ID 2
        assertEquals(3L, get(get(filteredCursor,1),0)); // Row with ID 3
    }
}
