package com.moneyfan.dsel;

import com.moneyfan.dsel.core.*;
import java.io.*;
import java.nio.BufferUnderflowException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import java.util.Arrays;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipEntry;

public enum D { // DSEL Operations Hub (D)
    ;
    // --- Core Join ---
    public static <F, S> Join<F, S> createJoin(F f, S s) { return Join.jn(f, s); }
    public static <F, S> F first(Join<F, S> j) { return j.f(); }
    public static <F, S> S second(Join<F, S> j) { return j.s(); }
    public static <F, S, FN> Join<FN, S> mapFirst(Join<F, S> j, Function<? super F, ? extends FN> m) { return j.mapFst(m); }
    public static <F, S, SN> Join<F, SN> mapSecond(Join<F, S> j, Function<? super S, ? extends SN> m) { return j.mapSnd(m); }
    public static <F, S, FN, SN> Join<FN, SN> mapJoin(Join<F,S> j, BiFunction<? super F,? super S,? extends Join<FN,SN>> bm) { return j.mapBoth(bm); }
    public static <F, S, FN, SN> Join<FN, SN> mapJoin(Join<F,S> j, Function<? super F,? extends FN> fm, Function<? super S,? extends SN> sm) { return j.mapBoth(fm, sm); }
    public static <F, S> Join<S, F> swapJoin(Join<F, S> j) { return j.swap(); }
    public static <F, S> boolean testJoin(Join<F, S> j, Predicate<? super Join<F, S>> p) { return p.test(j); } // Corrected: use p.test(j)

    public static <F, S> Join<F, S> jn(F f, S s) { return createJoin(f,s); }
    public static <F, S> F f(Join<F, S> j) { return first(j); }
    public static <F, S> S s(Join<F, S> j) { return second(j); }
    public static <F, S, FN> Join<FN, S> mf(Join<F, S> j, Function<? super F, ? extends FN> m) { return mapFirst(j,m); }
    public static <F, S, SN> Join<F, SN> ms(Join<F, S> j, Function<? super S, ? extends SN> m) { return mapSecond(j,m); }
    public static <F, S, FN, SN> Join<FN, SN> mb(Join<F,S> j, BiFunction<? super F,? super S,? extends Join<FN,SN>> bm) { return mapJoin(j,bm); }
    public static <F, S, FN, SN> Join<FN, SN> mb(Join<F,S> j, Function<? super F,? extends FN> fm, Function<? super S,? extends SN> sm) { return mapJoin(j,fm,sm); }
    public static <F, S> Join<S, F> sw(Join<F, S> j) { return swapJoin(j); }
    public static <F, S> boolean tst(Join<F, S> j, Predicate<? super Join<F, S>> p) { return testJoin(j,p); }

    // --- Conceptual Type Factories ---
    // Modified: Ensure these factories return instances of their respective interfaces
    public static <T> Series<T> createSeries(int size, Function<Integer, T> gen) { 
        if (size < 0) throw new IllegalArgumentException("Series size invalid: " + size); 
        return new Series<T>() {
            @Override public Integer f() { return size; }
            @Override public Function<Integer, T> s() { return gen; }
        };
    }
    public static ColumnMeta createColumnMeta(String n, TypeMemento t) { 
        return new ColumnMeta() {
            @Override public String f() { return n; }
            @Override public TypeMemento s() { return t; }
        };
    }
    public static RowVec createRowVec(int nCols, Function<Integer, Join<Object, Supplier<ColumnMeta>>> cellGen) { 
        return new RowVec() {
            @Override public Integer f() { return nCols; }
            @Override public Function<Integer, Join<Object, Supplier<ColumnMeta>>> s() { return cellGen; }
        };
    }
    public static Cursor createCursor(int nRows, Function<Integer, RowVec> rowGen) { 
        return new Cursor() {
            @Override public Integer f() { return nRows; }
            @Override public Function<Integer, RowVec> s() { return rowGen; }
        };
    }
    public static <T> Twin<T> createTwin(T t1, T t2) { 
        return new Twin<T>() {
            @Override public T f() { return t1; }
            @Override public T s() { return t2; }
        };
    }

    public static <T> Series<T> sr(int sz, Function<Integer, T> gen) { return createSeries(sz,gen); }
    public static ColumnMeta cm(String n, TypeMemento t) { return createColumnMeta(n,t); }
    public static RowVec rv(int sz, Function<Integer, Join<Object, Supplier<ColumnMeta>>> gen) { return createRowVec(sz,gen); }
    public static Cursor cr(int sz, Function<Integer, RowVec> gen) { return createCursor(sz,gen); }
    public static <T> Twin<T> tw(T t1, T t2) { return createTwin(t1,t2); }

    // --- Series Operations ---
    public static <T> int seriesSize(Series<T> s) { return s == null ? 0 : s.f(); }
    public static <T> T seriesGet(Series<T> s, int i) { return (s==null || i<0 || i>=s.f()) ? null : s.s().apply(i); }
    public static <T,R> Series<R> seriesMap(Series<T> s, Function<? super T,? extends R> m) { Objects.requireNonNull(s); Objects.requireNonNull(m); return sr(sz(s), i->m.apply(get(s,i))); }
    public static <T> Series<T> seriesFilter(Series<T> s, Predicate<? super T> p) {
        Objects.requireNonNull(s); Objects.requireNonNull(p);
        return new Series<>() { // Lazy caching filter
            private List<Integer> idxCache = null; private int sizeCache = -1;
            private void compute() { if (idxCache==null) { idxCache = new ArrayList<>(); for(int i=0;i<D.sz(s);i++) if(p.test(D.get(s,i))) idxCache.add(i); sizeCache=idxCache.size(); }}
            @Override public Integer f() { compute(); return sizeCache; }
            @Override public Function<Integer,T> s() { compute(); return i -> { if(i<0||i>=sizeCache) throw new IndexOutOfBoundsException(); return D.get(s, idxCache.get(i)); };}
        };
    }
    public static <T> Series<T> seriesHead(Series<T> s, int n) { Objects.requireNonNull(s); n=Math.max(0,n); return sr(Math.min(n,sz(s)), i->get(s,i)); }
    public static <T> Series<T> seriesTail(Series<T> s, int n) { Objects.requireNonNull(s); n=Math.max(0,n); int oSz=sz(s); int nSz=Math.min(n,oSz); return sr(nSz, i->get(s,oSz-nSz+i)); }
    public static <T> Series<T> seriesSkip(Series<T> s, int n) { 
        Objects.requireNonNull(s); 
        final int skipCount = Math.max(0, n); 
        final int originalSize = sz(s); 
        return sr(Math.max(0, originalSize - skipCount), i -> get(s, skipCount + i));
    }
    public static <T> List<T> seriesToList(Series<T> s) { Objects.requireNonNull(s); return seriesStream(s).collect(Collectors.toList()); }
    public static <T> Stream<T> seriesStream(Series<T> s) { Objects.requireNonNull(s); return IntStream.range(0,sz(s)).mapToObj(i->get(s,i)); }
    public static <T> void seriesForEach(Series<T> s, Consumer<? super T> c) { Objects.requireNonNull(s); seriesStream(s).forEach(c); }
    public static <T> T seriesFirst(Series<T> s) { Objects.requireNonNull(s); if(sz(s)==0) throw new java.util.NoSuchElementException(); return get(s,0); }
    public static <T> T seriesLast(Series<T> s) { Objects.requireNonNull(s); int z=sz(s); if(z==0) throw new java.util.NoSuchElementException(); return get(s,z-1); }

    public static <T> int sz(Series<T> s) { return seriesSize(s); }
    public static <T> T get(Series<T> s, int i) { return seriesGet(s,i); }
    public static <T,R> Series<R> map(Series<T> s, Function<? super T,? extends R> m) { return seriesMap(s,m); }
    public static <T> Series<T> flt(Series<T> s, Predicate<? super T> p) { return seriesFilter(s,p); }
    public static <T> List<T> ls(Series<T> s) { return seriesToList(s); }
    public static <T> Stream<T> st(Series<T> s) { return seriesStream(s); }
    public static <T> Series<T> hd(Series<T> s, int n) { return seriesHead(s,n); }
    public static <T> Series<T> tl(Series<T> s, int n) { return seriesTail(s,n); }
    public static <T> Series<T> sk(Series<T> s, int n) { return seriesSkip(s,n); }
    public static <T> void each(Series<T> s, Consumer<? super T> c) { seriesForEach(s,c); }
    public static <T> T fst(Series<T> s) { return seriesFirst(s); }
    public static <T> T lst(Series<T> s) { return seriesLast(s); }

    // --- RowVec Operations ---
    @SuppressWarnings("unchecked")
    public static Join<Object,Supplier<ColumnMeta>> rowVecGetCell(RowVec rv, int colIdx) { return (Join<Object,Supplier<ColumnMeta>>)get(rv,colIdx); }
    public static Object rowVecGetValue(RowVec rv, int colIdx) { Join<Object,Supplier<ColumnMeta>> c=cell(rv,colIdx); return c!=null?c.f():null;}
    public static String rowVecGetColName(RowVec rv, int colIdx) { ColumnMeta m=cell(rv,colIdx).s().get(); return m!=null?m.f():null; }
    public static TypeMemento rowVecGetColType(RowVec rv, int colIdx) { ColumnMeta m=cell(rv,colIdx).s().get(); return m!=null?m.s():null; }

    public static Join<Object,Supplier<ColumnMeta>> cell(RowVec rv, int colIdx) { return rowVecGetCell(rv,colIdx); }
    public static Object get(RowVec rv, int colIdx) { return rowVecGetValue(rv,colIdx); }
    public static String colName(RowVec rv, int colIdx) { return rowVecGetColName(rv,colIdx); }
    public static TypeMemento colType(RowVec rv, int colIdx) { return rowVecGetColType(rv,colIdx); }

    // --- Cursor Operations ---
    public static RowVec cursorGetRow(Cursor c, int rowIdx) { return get(c,rowIdx); }
    public static Cursor cursorMapRows(Cursor c, Function<RowVec,RowVec> m) { return (Cursor)map(c,m); }
    public static Cursor cursorFilterRows(Cursor c, Predicate<RowVec> p) { return (Cursor)flt(c,p); }

    public static RowVec get(Cursor c, int rowIdx) { return cursorGetRow(c,rowIdx); }
    public static Cursor mapRow(Cursor c, Function<RowVec,RowVec> m) { return cursorMapRows(c,m); }
    public static Cursor fltRow(Cursor c, Predicate<RowVec> p) { return cursorFilterRows(c,p); }

    // --- ISAM File Handling ---
    public static final String META_SFX = ".meta"; public static final String DATA_SFX = ".data";

    public record IsamFileMetadata(List<ColumnMeta> schema, long recordCount, int recordByteLength) {
        public IsamFileMetadata(List<ColumnMeta> schema, long recordCount) {
            this(List.copyOf(Objects.requireNonNull(schema)), recordCount, schema.stream().mapToInt(cm -> cm.s().getFixedSize()).sum());
            if (this.recordByteLength <= 0 && !schema.isEmpty() && schema.stream().anyMatch(cm -> cm.s().getFixedSize() == -1))
                throw new IllegalArgumentException("ISAM requires all columns to have fixed size for this implementation.");
            if (this.recordByteLength <= 0 && !schema.isEmpty() && schema.stream().allMatch(cm -> cm.s().getFixedSize() != -1))
                throw new IllegalArgumentException("Non-positive record byte length with fixed-size schema: " + this.recordByteLength);
        }
        public void write(DataOutputStream out) throws IOException {
            out.writeInt(schema.size());
            for (ColumnMeta cm : schema) { out.writeUTF(cm.f()); out.writeUTF(cm.s().getTypeName()); out.writeInt(cm.s().getFixedSize()); }
            out.writeLong(recordCount); out.writeInt(recordByteLength);
        }
        public static IsamFileMetadata read(DataInputStream in) throws IOException {
            int schemaSize = in.readInt(); List<ColumnMeta> schema = new ArrayList<>(schemaSize);
            for (int i = 0; i < schemaSize; i++) {
                String n = in.readUTF(); String tn = in.readUTF(); int fs = in.readInt();
                TypeMemento t;
                // Check if it's a custom-sized string or a basic type
                if (tn.startsWith(TypeMemento.CUSTOM_STRING_PREFIX)) {
                    t = TypeMemento.customString(fs);
                } else {
                    t = TypeMemento.Basic.fromTypeName(tn);
                    // If the basic type's fixed size doesn't match the stored one,
                    // it means the stored metadata has a custom size for a "basic" type.
                    // In this case, create a CustomType to preserve the stored fixed size.
                    if (t.getFixedSize() != fs && fs != -1) {
                        System.err.println("Warning: Mismatch in stored fixed size ("+ fs +") and enum fixed size ("+ t.getFixedSize() +") for type " + tn + ". Using stored fixed size.");
                        t = new TypeMemento.CustomType(tn, fs);
                    }
                }
                schema.add(D.cm(n, t));
            }
            return new IsamFileMetadata(schema, in.readLong(), in.readInt());
        }
    }

    public static class IsamCursor implements Cursor {
        private final RandomAccessFile dataFile; private final IsamFileMetadata metadata;
        private final String pathBase;

        public IsamCursor(String pathBase) throws IOException {
            this.pathBase = pathBase;
            try (DataInputStream metaIn = new DataInputStream(new FileInputStream(pathBase + META_SFX))) {
                this.metadata = IsamFileMetadata.read(metaIn);
            }
            this.dataFile = new RandomAccessFile(pathBase + DATA_SFX, "r");
        }
        @Override public Integer f() { return (int) metadata.recordCount(); } // Assuming count fits int for Series
        @Override public Function<Integer, RowVec> s() {
            return rowIndex -> {
                if (rowIndex < 0 || rowIndex >= metadata.recordCount()) throw new IndexOutOfBoundsException();
                try {
                    byte[] rowBytes = new byte[metadata.recordByteLength()];
                    synchronized(dataFile) { // Synchronize access if shared or make RAFile thread-local
                        dataFile.seek((long)rowIndex * metadata.recordByteLength());
                        dataFile.readFully(rowBytes);
                    }
                    ByteBuffer bb = ByteBuffer.wrap(rowBytes).order(ByteOrder.BIG_ENDIAN);
                    return rv(metadata.schema().size(), colIndex -> {
                        ColumnMeta cm = metadata.schema().get(colIndex);
                        Object val = readValueFromBuffer(bb, cm.s());
                        return jn(val, () -> cm);
                    });
                } catch (IOException e) { throw new UncheckedIOException(e); }
            };
        }
        public void close() throws IOException { if (dataFile != null) dataFile.close(); }
    }

    public static void writeIsam(String pathBase, Cursor data) throws IOException {
        if (sz(data) == 0) { // Handle empty cursor
            try (DataOutputStream metaOut = new DataOutputStream(new FileOutputStream(pathBase + META_SFX))) {
                new IsamFileMetadata(Collections.emptyList(), 0, 0).write(metaOut);
            } // Create empty data file implicitly or explicitly if needed
            try (FileOutputStream dataOut = new FileOutputStream(pathBase + DATA_SFX)) { /* empty */ }
            return;
        }
        RowVec firstRow = get(data, 0);
        List<ColumnMeta> schema = ls(firstRow).stream().map(cell -> cell.s().get()).collect(Collectors.toList());
        IsamFileMetadata meta = new IsamFileMetadata(schema, sz(data));

        try (DataOutputStream metaOut = new DataOutputStream(new FileOutputStream(pathBase + META_SFX));
             RandomAccessFile dataOut = new RandomAccessFile(pathBase + DATA_SFX, "rw")) {
            dataOut.setLength(0); meta.write(metaOut);
            ByteBuffer bb = ByteBuffer.allocate(meta.recordByteLength()).order(ByteOrder.BIG_ENDIAN);
            for (int i = 0; i < sz(data); i++) {
                RowVec row = get(data, i); bb.clear();
                for (int j = 0; j < sz(row); j++) writeValueToBuffer(bb, cell(row,j).f(), schema.get(j).s());
                dataOut.write(bb.array());
            }
        }
    }

    // --- ISAM Value Serializers/Deserializers ---
    public static void writeValueToBuffer(ByteBuffer bb, Object val, TypeMemento type) {
        switch (type.getTypeName()) { // Use getTypeName() for switch
            case "Boolean": bb.put((byte)((Boolean)val ? 1 : 0)); break;
            case "Byte": bb.put((Byte)val); break;
            case "Short": bb.putShort((Short)val); break;
            case "Integer": bb.putInt((Integer)val); break;
            case "Long": bb.putLong((Long)val); break;
            case "Float": bb.putFloat((Float)val); break;
            case "Double": bb.putDouble((Double)val); break;
            case "Char": bb.putChar((Character)val); break;
            case "String": // This case handles TypeMemento.Basic.STRING
            case TypeMemento.CUSTOM_STRING_PREFIX + "256": // Example for a specific custom string size
                // Fall through to handle custom strings based on their fixed size
            case "BinaryBlob": // This case handles TypeMemento.Basic.BINARY_BLOB
                // Fall through to handle custom binary blobs based on their fixed size
                byte[] bytes;
                if (type.getTypeName().startsWith(TypeMemento.CUSTOM_STRING_PREFIX) || type.getTypeName().equals("String")) {
                    bytes = ((String)val).getBytes(StandardCharsets.UTF_8);
                } else if (type.getTypeName().equals("BinaryBlob")) {
                    bytes = (byte[])val;
                } else {
                    throw new IllegalArgumentException("Unsupported type for fixed-size ISAM: " + type.getTypeName());
                }

                int len = Math.min(bytes.length, type.getFixedSize());
                if(type.getFixedSize() <=0) throw new IllegalArgumentException(type.getTypeName() + " requires fixed size in this ISAM: " + type.getTypeName());
                bb.put(bytes, 0, len); for(int i=len; i<type.getFixedSize(); i++) bb.put((byte)0); // Pad
                break;
            default: throw new IllegalArgumentException("Unsupported type for fixed-size ISAM: " + type.getTypeName());
        }
    }
    public static Object readValueFromBuffer(ByteBuffer bb, TypeMemento type) {
        try {
            switch (type.getTypeName()) { // Use getTypeName() for switch
                case "Boolean": return bb.get() == 1;
                case "Byte": return bb.get();
                case "Short": return bb.getShort();
                case "Integer": return bb.getInt();
                case "Long": return bb.getLong();
                case "Float": return bb.getFloat();
                case "Double": return bb.getDouble();
                case "Char": return bb.getChar();
                case "String":
                case TypeMemento.CUSTOM_STRING_PREFIX + "256": // Example for a specific custom string size
                    if(type.getFixedSize() <=0) throw new IllegalArgumentException("String requires fixed size in this ISAM: " + type.getTypeName());
                    byte[] strBytes = new byte[type.getFixedSize()]; bb.get(strBytes);
                    int actualLen = 0; while(actualLen < strBytes.length && strBytes[actualLen] != 0) actualLen++; // Find null terminator
                    return new String(strBytes, 0, actualLen, StandardCharsets.UTF_8);
                case "BinaryBlob":
                    if(type.getFixedSize() <=0) throw new IllegalArgumentException("BINARY_BLOB requires fixed size in this ISAM: " + type.getTypeName());
                    byte[] blobBytes = new byte[type.getFixedSize()]; bb.get(blobBytes);
                    return blobBytes; // Return full fixed-size buffer, or trim if null-termination is convention
                default: throw new IllegalArgumentException("Unsupported type for ISAM: " + type.getTypeName());
            }
        } catch (BufferUnderflowException e) {
            throw new RuntimeException("Buffer underflow while reading type " + type.getTypeName(), e);
        }
    }

    // --- CSV Utilities ---
    public static RowVec parseCsvLine(String line, List<ColumnMeta> schema, String delimiter) {
        String[] parts = line.split(Objects.requireNonNull(delimiter, "Delimiter cannot be null"), -1); // -1 to keep trailing empty strings
        if (parts.length != schema.size()) throw new IllegalArgumentException("CSV line parts ("+parts.length+") != schema size ("+schema.size()+"): " + line);
        return rv(schema.size(), colIdx -> {
            ColumnMeta cm = schema.get(colIdx);
            Object val = parseStringValue(parts[colIdx], cm.s());
            return jn(val, () -> cm);
        });
    }

    public static Object parseStringValue(String sVal, TypeMemento type) {
        try {
            // Use getTypeName() for switch
            return switch(type.getTypeName()) {
                case "Boolean": yield Boolean.parseBoolean(sVal);
                case "Byte": yield Byte.parseByte(sVal);
                case "Short": yield Short.parseShort(sVal);
                case "Integer": yield Integer.parseInt(sVal);
                case "Long": yield Long.parseLong(sVal);
                case "Float": yield Float.parseFloat(sVal);
                case "Double": yield Double.parseDouble(sVal);
                case "Char": yield (sVal.isEmpty() ? '\0' : sVal.charAt(0));
                case "String": // Handles TypeMemento.Basic.STRING and custom fixed-size strings
                case TypeMemento.CUSTOM_STRING_PREFIX + "256": // Example for a specific custom string size
                    yield sVal;
                default: throw new IllegalArgumentException("Cannot parse CSV string to type: " + type.getTypeName());
            };
        } catch (NumberFormatException e) {
            System.err.println("Error parsing '" + sVal + "' as " + type.getTypeName() + ": " + e.getMessage() + ". Returning null or default.");
            // Default values on parse error
            return switch(type.getTypeName()) {
                 case "Boolean": yield false; case "Byte": yield (byte)0; case "Short": yield (short)0; case "Integer": yield 0;
                 case "Long": yield 0L; case "Float": yield 0.0f; case "Double": yield 0.0; case "Char": yield '\0';
                 case "String": case TypeMemento.CUSTOM_STRING_PREFIX + "256": yield ""; default: yield null;
            };
        }
    }

    public static Cursor readCsv(String filePath, List<ColumnMeta> schema, String delimiter, boolean skipHeader) throws IOException {
        Path path = Paths.get(filePath);
        try (Stream<String> lines = Files.lines(path)) {
            List<RowVec> rows = lines.skip(skipHeader ? 1 : 0)
                                     .map(line -> parseCsvLine(line, schema, delimiter))
                                     .filter(Objects::nonNull) // In case parseCsvLine can return null
                                     .collect(Collectors.toList());
            return cr(rows.size(), rows::get);
        }
    }

    public static void csvToIsam(String csvPath, String isamPathBase, List<ColumnMeta> schema, String delimiter, boolean skipHeader) throws IOException {
        Cursor csvCursor = readCsv(csvPath, schema, delimiter, skipHeader);
        writeIsam(isamPathBase, csvCursor);
    }
// --- Binance Data Utilities ---
    public static List<ColumnMeta> getBinanceKlineSchema() {
        return Arrays.asList(
            cm("Open_time", TypeMemento.Basic.LONG), // Changed to LONG to match ISAM handling
            cm("Open", TypeMemento.Basic.DOUBLE),
            cm("High", TypeMemento.Basic.DOUBLE),
            cm("Low", TypeMemento.Basic.DOUBLE),
            cm("Close", TypeMemento.Basic.DOUBLE),
            cm("Volume", TypeMemento.Basic.DOUBLE),
            cm("Close_time", TypeMemento.Basic.LONG), // Changed to LONG to match ISAM handling
            cm("Quote_asset_volume", TypeMemento.Basic.DOUBLE),
            cm("Number_of_trades", TypeMemento.Basic.INTEGER),
            cm("Taker_buy_base_asset_volume", TypeMemento.Basic.DOUBLE),
            cm("Taker_buy_quote_asset_volume", TypeMemento.Basic.DOUBLE),
            cm("Ignore", TypeMemento.customString(256)) // Use customString for fixed-size string
        );
    }

    public static void binanceCsvToIsam(String csvPath, String isamPathBase, boolean skipHeader) throws IOException {
        List<ColumnMeta> schema = getBinanceKlineSchema();
        csvToIsamStreaming(csvPath, isamPathBase, schema, ",", skipHeader);
    }

    public static void binanceArchiveToIsam(String zipPath, String isamPathBase, String csvEntryName, boolean skipHeader) throws IOException {
        List<ColumnMeta> schema = getBinanceKlineSchema();
        Path tempCsvPath = Files.createTempFile("binance_temp", ".csv");
        try (ZipInputStream zis = new ZipInputStream(new FileInputStream(zipPath))) {
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                if (entry.getName().equals(csvEntryName)) {
                    try (FileOutputStream fos = new FileOutputStream(tempCsvPath.toFile())) {
                        byte[] buffer = new byte[1024];
                        int len;
                        while ((len = zis.read(buffer)) > 0) {
                            fos.write(buffer, 0, len);
                        }
                    }
                    break;
                }
                zis.closeEntry();
            }
        }
        try {
            csvToIsamStreaming(tempCsvPath.toString(), isamPathBase, schema, ",", skipHeader);
        } finally {
            Files.deleteIfExists(tempCsvPath);
        }
    }

    public static void csvToIsamStreaming(String csvPath, String isamPathBase, List<ColumnMeta> schema, String delimiter, boolean skipHeader) throws IOException {
        Path path = Paths.get(csvPath);
        long recordCount = 0;
        try (DataOutputStream metaOut = new DataOutputStream(new FileOutputStream(isamPathBase + META_SFX));
             RandomAccessFile dataOut = new RandomAccessFile(isamPathBase + DATA_SFX, "rw");
             BufferedReader reader = Files.newBufferedReader(path)) {
            dataOut.setLength(0);
            if (skipHeader) reader.readLine(); // Skip header
            String line = reader.readLine();
            if (line == null) {
                // Handle empty CSV
                new IsamFileMetadata(Collections.emptyList(), 0, 0).write(metaOut);
                return;
            }
            // Ensure String columns in schema have a fixed size defined for ISAM.
            List<ColumnMeta> fixedSchema = schema.stream().map(cm -> {
                if (cm.s().getFixedSize() == -1 && cm.s().getTypeName().equals(TypeMemento.Basic.STRING.getTypeName())) {
                    // Provide a sensible default for strings if not already fixed.
                    return D.cm(cm.f(), TypeMemento.customString(256)); // Using new factory method
                }
                return cm;
            }).collect(Collectors.toList());

            RowVec firstRow = parseCsvLine(line, fixedSchema, delimiter);
            IsamFileMetadata meta = new IsamFileMetadata(fixedSchema, 0); // Temporary count, will update later
            meta.write(metaOut); // Write placeholder metadata
            ByteBuffer bb = ByteBuffer.allocate(meta.recordByteLength()).order(ByteOrder.BIG_ENDIAN);
            // Process first row
            bb.clear();
            for (int j = 0; j < sz(firstRow); j++) writeValueToBuffer(bb, cell(firstRow, j).f(), fixedSchema.get(j).s());
            dataOut.write(bb.array());
            recordCount++;
            // Process remaining rows in a single pass
            while ((line = reader.readLine()) != null) {
                RowVec row = parseCsvLine(line, fixedSchema, delimiter);
                bb.clear();
                for (int j = 0; j < sz(row); j++) writeValueToBuffer(bb, cell(row, j).f(), fixedSchema.get(j).s());
                dataOut.write(bb.array());
                recordCount++;
            }
            // Update metadata with actual record count
            try (RandomAccessFile metaRaf = new RandomAccessFile(isamPathBase + META_SFX, "rw")) {
                metaRaf.seek(0);
                new IsamFileMetadata(fixedSchema, recordCount).write(new DataOutputStream(new FileOutputStream(metaRaf.getFD())));
            }
        }
    }

    public static class TimeSeriesIsamCursor implements Cursor {
        private final RandomAccessFile dataFile;
        private final RandomAccessFile indexFile;
        private final IsamFileMetadata metadata;
        private final String pathBase;
        private final long indexCount;
        private static final int MAX_IN_MEMORY_INDEX = 1000000; // Limit to prevent memory issues
        private final boolean useDiskIndex; // Flag to use disk-based search for large indexes
        private final long[] timestamps; // Only used if not using disk index
        private final long[] offsets; // Only used if not using disk index

        public TimeSeriesIsamCursor(String pathBase) throws IOException {
            this.pathBase = pathBase;
            try (DataInputStream metaIn = new DataInputStream(new FileInputStream(pathBase + META_SFX))) {
                this.metadata = IsamFileMetadata.read(metaIn);
            }
            this.dataFile = new RandomAccessFile(pathBase + DATA_SFX, "r");
            // Check if index file exists, if not, create an empty array/handle missing index
            Path indexPath = Paths.get(pathBase + DATA_SFX + ".idx");
            if (Files.exists(indexPath)) {
                this.indexFile = new RandomAccessFile(indexPath.toFile(), "r");
                this.indexCount = indexFile.length() / 16; // 8 bytes for timestamp, 8 for offset
                this.useDiskIndex = indexCount > MAX_IN_MEMORY_INDEX;
                if (!useDiskIndex) {
                    timestamps = new long[(int) indexCount];
                    offsets = new long[(int) indexCount];
                    indexFile.seek(0);
                    for (int i = 0; i < indexCount; i++) {
                        timestamps[i] = indexFile.readLong();
                        offsets[i] = indexFile.readLong();
                    }
                } else {
                    timestamps = new long[0];
                    offsets = new long[0];
                }
            } else {
                this.indexFile = null; // No index file
                this.indexCount = 0;
                this.useDiskIndex = false;
                this.timestamps = new long[0];
                this.offsets = new long[0];
            }
        }

        @Override
        public Integer f() {
            return (int) metadata.recordCount();
        }

        @Override
        public Function<Integer, RowVec> s() {
            return rowIndex -> {
                if (rowIndex < 0 || rowIndex >= metadata.recordCount()) throw new IndexOutOfBoundsException();
                try {
                    byte[] rowBytes = new byte[metadata.recordByteLength()];
                    synchronized (dataFile) {
                        dataFile.seek((long) rowIndex * metadata.recordByteLength());
                        dataFile.readFully(rowBytes);
                    }
                    ByteBuffer bb = ByteBuffer.wrap(rowBytes).order(ByteOrder.BIG_ENDIAN);
                    return rv(metadata.schema().size(), colIndex -> {
                        ColumnMeta cm = metadata.schema().get(colIndex);
                        Object val = readValueFromBuffer(bb, cm.s());
                        return jn(val, () -> cm);
                    });
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            };
        }

        public int findRowIndexByTimestamp(long timestamp) throws IOException {
            if (indexFile == null || indexCount == 0) return -1; // No index to search
            if (useDiskIndex) {
                return diskBinarySearch(timestamp);
            } else {
                // Binary search on in-memory timestamps array
                int left = 0, right = timestamps.length - 1;
                while (left <= right) {
                    int mid = left + (right - left) / 2;
                    if (timestamps[mid] == timestamp) return mid;
                    else if (timestamps[mid] < timestamp) left = mid + 1;
                    else right = mid - 1;
                }
                return -1; // Not found
            }
        }

        public Join<Integer, Integer> findRowIndexRangeByTimestamp(long startTimestamp, long endTimestamp) throws IOException {
            if (indexFile == null || indexCount == 0) return jn(-1,-1); // No index to search

            if (useDiskIndex) {
                int startIdx = diskBinarySearchNearest(startTimestamp, true);
                int endIdx = diskBinarySearchNearest(endTimestamp, false);
                return jn(startIdx, endIdx);
            } else {
                // Binary search for range in memory
                int startIdx = -1, endIdx = -1;
                int left = 0, right = timestamps.length - 1;
                // Find start index (first >= startTimestamp)
                while (left <= right) {
                    int mid = left + (right - left) / 2;
                    if (timestamps[mid] >= startTimestamp) {
                        startIdx = mid;
                        right = mid - 1;
                    } else {
                        left = mid + 1;
                    }
                }
                // Find end index (last <= endTimestamp)
                left = 0;
                right = timestamps.length - 1;
                while (left <= right) {
                    int mid = left + (right - left) / 2;
                    if (timestamps[mid] <= endTimestamp) {
                        endIdx = mid;
                        left = mid + 1;
                    } else {
                        right = mid - 1;
                    }
                }
                return jn(startIdx, endIdx);
            }
        }

        private int diskBinarySearch(long timestamp) throws IOException {
            long left = 0, right = indexCount - 1;
            while (left <= right) {
                long mid = left + (right - left) / 2;
                long midTimestamp = readTimestampAtIndex(mid);
                if (midTimestamp == timestamp) return (int) mid;
                else if (midTimestamp < timestamp) left = mid + 1;
                else right = mid - 1;
            }
            return -1; // Not found
        }

        private int diskBinarySearchNearest(long timestamp, boolean findFirst) throws IOException {
            long left = 0, right = indexCount - 1;
            long nearestIdx = -1;
            while (left <= right) {
                long mid = left + (right - left) / 2;
                long midTimestamp = readTimestampAtIndex(mid);
                if (midTimestamp == timestamp) return (int) mid;
                else if (midTimestamp < timestamp) {
                    if (findFirst) nearestIdx = mid; // Update nearest for first occurrence
                    left = mid + 1;
                } else {
                    if (!findFirst) nearestIdx = mid; // Update nearest for last occurrence
                    right = mid - 1;
                }
            }
            return (int) nearestIdx;
        }

        private long readTimestampAtIndex(long index) throws IOException {
            if (indexFile == null) throw new IOException("Index file not available.");
            synchronized (indexFile) {
                indexFile.seek(index * 16); // 8 bytes timestamp + 8 bytes offset
                return indexFile.readLong();
            }
        }

        public void close() throws IOException {
            if (dataFile != null) dataFile.close();
            if (indexFile != null) indexFile.close();
        }
    }

    public static void writeIsamWithTimeIndex(String pathBase, Cursor data, int timestampColIdx) throws IOException {
        if (sz(data) == 0) {
            try (DataOutputStream metaOut = new DataOutputStream(new FileOutputStream(pathBase + META_SFX))) {
                new IsamFileMetadata(Collections.emptyList(), 0, 0).write(metaOut);
            }
            try (FileOutputStream dataOut = new FileOutputStream(pathBase + DATA_SFX)) { /* empty */ }
            try (FileOutputStream idxOut = new FileOutputStream(pathBase + DATA_SFX + ".idx")) { /* empty */ }
            return;
        }
        RowVec firstRow = get(data, 0);
        List<ColumnMeta> schema = ls(firstRow).stream().map(cell -> cell.s().get()).collect(Collectors.toList());

        // Ensure String columns in schema have a fixed size defined for ISAM.
        List<ColumnMeta> fixedSchema = schema.stream().map(cm -> {
            if (cm.s().getFixedSize() == -1 && cm.s().getTypeName().equals(TypeMemento.Basic.STRING.getTypeName())) {
                // Provide a sensible default fixed size for strings.
                // This might need to be adjusted based on the actual max length of strings.
                return D.cm(cm.f(), TypeMemento.customString(256)); // Using new factory method
            }
            return cm;
        }).collect(Collectors.toList());

        IsamFileMetadata meta = new IsamFileMetadata(fixedSchema, sz(data));

        List<Join<Long, Long>> timestampOffsets = new ArrayList<>();
        try (DataOutputStream metaOut = new DataOutputStream(new FileOutputStream(pathBase + META_SFX));
             RandomAccessFile dataOut = new RandomAccessFile(pathBase + DATA_SFX, "rw")) {
            dataOut.setLength(0);
            meta.write(metaOut);
            ByteBuffer bb = ByteBuffer.allocate(meta.recordByteLength()).order(ByteOrder.BIG_ENDIAN);
            for (int i = 0; i < sz(data); i++) {
                long offset = dataOut.getFilePointer();
                RowVec row = get(data, i);
                bb.clear();
                for (int j = 0; j < sz(row); j++) {
                    Object val = cell(row, j).f();
                    if (j == timestampColIdx && val instanceof Long) {
                        timestampOffsets.add(jn((Long) val, offset));
                    }
                    writeValueToBuffer(bb, val, fixedSchema.get(j).s());
                }
                dataOut.write(bb.array());
            }
        }
        // Write index file
        timestampOffsets.sort((a, b) -> Long.compare(a.f(), b.f()));
        try (DataOutputStream idxOut = new DataOutputStream(new FileOutputStream(pathBase + DATA_SFX + ".idx"))) {
            for (Join<Long, Long> tsOffset : timestampOffsets) {
                idxOut.writeLong(tsOffset.f()); // Timestamp
                idxOut.writeLong(tsOffset.s()); // Offset
            }
        }
    }
}
