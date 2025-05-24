package com.moneyfan.dsel;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

//#################################################################################
//### 1. CORE DSEL TYPE INTERFACES (as static inner interfaces of D)
//#################################################################################

interface Join<F, S> {
    F f();
    S s();

    static <F_TYPE, S_TYPE> Join<F_TYPE, S_TYPE> jn(F_TYPE f, S_TYPE s) {
        return new D.ImmutableJoinRecord<>(f, s);
    }

    default <FN> Join<FN, S> mapFst(Function<? super F, ? extends FN> m) {
        Objects.requireNonNull(m);
        return jn(m.apply(f()), s());
    }

    default <SN> Join<F, SN> mapSnd(Function<? super S, ? extends SN> m) {
        Objects.requireNonNull(m);
        return jn(f(), m.apply(s()));
    }

    default <FN, SN> Join<FN, SN> mapBoth(BiFunction<? super F, ? super S, ? extends Join<FN, SN>> bm) {
        Objects.requireNonNull(bm);
        return bm.apply(f(), s());
    }

    default <FN, SN> Join<FN, SN> mapBoth(Function<? super F, ? extends FN> fm, Function<? super S, ? extends SN> sm) {
        Objects.requireNonNull(fm);
        Objects.requireNonNull(sm);
        return jn(fm.apply(f()), sm.apply(s()));
    }

    default Join<S, F> swap() {
        return jn(s(), f());
    }

    default boolean test(Predicate<? super Join<F, S>> p) {
        Objects.requireNonNull(p);
        return p.test(this);
    }
}

interface TypeMemento {
    String getTypeName();
    int getFixedSize();

    enum Basic implements TypeMemento {
        BOOLEAN("Boolean", 1),
        BYTE("Byte", 1),
        SHORT("Short", 2),
        INTEGER("Integer", 4),
        LONG("Long", 8),
        FLOAT("Float", 4),
        DOUBLE("Double", 8),
        CHAR("Char", 2),
        STRING("String", -1),
        BINARY_BLOB("BinaryBlob", -1),
        OBJECT("Object", -1),
        JOIN("Join", -1),
        SERIES("Series", -1),
        ROWVEC("RowVec", -1),
        CURSOR("Cursor", -1),
        TWIN("Twin", -1),
        CUSTOM("Custom", -1),
        UNKNOWN("Unknown", -1); // Fallback for initial deduction

        private final String typeName;
        private final int fixedSize;

        Basic(String typeName, int fixedSize) {
            this.typeName = typeName;
            this.fixedSize = fixedSize;
        }

        @Override public String getTypeName() { return typeName; }
        @Override public int getFixedSize() { return fixedSize; }

        public static TypeMemento fromTypeName(String name) {
            for (Basic b : values()) {
                if (b.getTypeName().equals(name)) return b;
            }
            throw new IllegalArgumentException("Unknown TypeMemento basic name: " + name);
        }
    }
}

interface Series<T> extends Join<Integer, Function<Integer, T>> {}
interface ColumnMeta extends Join<String, TypeMemento> {}
interface RowVec extends Series<Join<Object, Supplier<ColumnMeta>>> {}
interface Cursor extends Series<RowVec> {}
interface Twin<T> extends Join<T,T> {}


//#################################################################################
//### 2. D (Data Science Expression Language) Omnibus Enum & Operations Hub
//#################################################################################
public enum D {
    ; // Enum instances are not used; this is a static utility class pattern

    // --- ImmutableJoinRecord (helper for Join.jn) ---
    record ImmutableJoinRecord<F, S>(F f, S s) implements Join<F, S> {
        @Override
        public String toString() {
            return "jn(" + f + ", " + s + ")";
        }
    }

    //#################################################################################
    //### 3. TYPE EVIDENCE for CSV Type Deduction
    //#################################################################################
    static class TypeEvidence {
        long longValueCount = 0;
        long doubleValueCount = 0;
        long booleanValueCount = 0;
        long stringValueCount = 0;
        long emptyCount = 0;
        int maxStrLength = 0;
        String firstNonEmptySample = null;

        public void assess(String fieldValue) {
            if (fieldValue == null || fieldValue.isEmpty()) {
                emptyCount++;
                return;
            }
            if (firstNonEmptySample == null) {
                firstNonEmptySample = fieldValue;
            }
            maxStrLength = Math.max(maxStrLength, fieldValue.length());

            // Attempt to parse in order: boolean, long, double. If all fail, it's a string.
            String lowerField = fieldValue.toLowerCase();
            if ("true".equals(lowerField) || "false".equals(lowerField)) {
                booleanValueCount++;
                return; // Considered boolean for now
            }
            try {
                Long.parseLong(fieldValue);
                longValueCount++;
                return;
            } catch (NumberFormatException ignoredL) {
                // Not a long
            }
            try {
                Double.parseDouble(fieldValue);
                doubleValueCount++;
                return;
            } catch (NumberFormatException ignoredD) {
                // Not a double
            }
            stringValueCount++; // Fallback to string
        }

        public TypeMemento deduceFinalType() {
            long totalNonEmpty = longValueCount + doubleValueCount + booleanValueCount + stringValueCount;

            if (totalNonEmpty == 0) { // All fields were empty or null
                return D.fsString(Math.max(1, maxStrLength)); // Default to string for empty, with observed max length
            }

            if (stringValueCount > 0) { // If any field strictly parsed as string, whole column is string
                return D.fsString(maxStrLength);
            }
            if (doubleValueCount > 0) { // If no strings, but doubles are present, it's double
                return TypeMemento.Basic.DOUBLE;
            }
            if (longValueCount > 0) { // If no strings/doubles, but longs are present, it's long
                return TypeMemento.Basic.LONG;
            }
            if (booleanValueCount > 0) { // Only booleans and empty strings
                return TypeMemento.Basic.BOOLEAN;
            }
            // Should not be reached if totalNonEmpty > 0
            return D.fsString(Math.max(1, maxStrLength)); // Fallback
        }
        
        @Override
        public String toString() {
            return "TypeEvidence{" +
                   "L:" + longValueCount +
                   ", D:" + doubleValueCount +
                   ", B:" + booleanValueCount +
                   ", S:" + stringValueCount +
                   ", E:" + emptyCount +
                   ", maxLen:" + maxStrLength +
                   (firstNonEmptySample != null ? ", sample:'" + firstNonEmptySample + "'" : "") +
                   '}';
        }
    }


    // --- Core Join Operations ---
    public static <F, S> Join<F, S> createJoin(F f, S s) { return Join.jn(f, s); }
    public static <F, S> F first(Join<F, S> j) { return j.f(); }
    public static <F, S> S second(Join<F, S> j) { return j.s(); }
    public static <F, S, FN> Join<FN, S> mapFirst(Join<F, S> j, Function<? super F, ? extends FN> m) { return j.mapFst(m); }
    public static <F, S, SN> Join<F, SN> mapSecond(Join<F, S> j, Function<? super S, ? extends SN> m) { return j.mapSnd(m); }
    public static <F, S, FN, SN> Join<FN, SN> mapJoin(Join<F,S> j, BiFunction<? super F,? super S,? extends Join<FN,SN>> bm) { return j.mapBoth(bm); }
    public static <F, S, FN, SN> Join<FN, SN> mapJoin(Join<F,S> j, Function<? super F,? extends FN> fm, Function<? super S,? extends SN> sm) { return j.mapBoth(fm, sm); }
    public static <F, S> Join<S, F> swapJoin(Join<F, S> j) { return j.swap(); }
    public static <F, S> boolean testJoin(Join<F, S> j, Predicate<? super Join<F, S>> p) { return j.test(p); }

    // Terse names for Join Operations
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
    public static <T> Series<T> createSeries(int size, Function<Integer, T> generator) {
        if (size < 0) throw new IllegalArgumentException("Series size cannot be negative: " + size);
        return (Series<T>) jn(size, generator);
    }
    public static ColumnMeta createColumnMeta(String name, TypeMemento type) {
        Objects.requireNonNull(name, "Column name cannot be null");
        Objects.requireNonNull(type, "Column type cannot be null");
        return (ColumnMeta) jn(name, type);
    }
    public static RowVec createRowVec(int numCols, Function<Integer, Join<Object, Supplier<ColumnMeta>>> cellGenerator) {
        return (RowVec) createSeries(numCols, cellGenerator);
    }
    public static Cursor createCursor(int numRows, Function<Integer, RowVec> rowGenerator) {
        return (Cursor) createSeries(numRows, rowGenerator);
    }
    public static <T> Twin<T> createTwin(T first, T second) {
        return (Twin<T>) jn(first, second);
    }

    // Terse names for Conceptual Type Factories
    public static <T> Series<T> sr(int size, Function<Integer, T> gen) { return createSeries(size,gen); }
    public static ColumnMeta cm(String n, TypeMemento t) { return createColumnMeta(n,t); }
    public static RowVec rv(int sz, Function<Integer, Join<Object, Supplier<ColumnMeta>>> gen) { return createRowVec(sz,gen); }
    public static Cursor cr(int sz, Function<Integer, RowVec> gen) { return createCursor(sz,gen); }
    public static <T> Twin<T> tw(T t1, T t2) { return createTwin(t1,t2); }


    // --- Fixed-Size TypeMementos for ISAM & Deduced CSV Types ---
    private interface FixedSizeTypeMemento extends TypeMemento {} // Marker interface

    public static TypeMemento fsString(int length) {
        if (length <= 0) throw new IllegalArgumentException("Fixed string length must be positive: " + length);
        return new FixedSizeTypeMemento() {
            @Override public String getTypeName() { return TypeMemento.Basic.STRING.getTypeName(); }
            @Override public int getFixedSize() { return length; }
            @Override public String toString() { return "STRING(" + length + ")";}
        };
    }
    public static TypeMemento fsBinaryBlob(int length) {
        if (length <= 0) throw new IllegalArgumentException("Fixed blob length must be positive: " + length);
        return new FixedSizeTypeMemento() {
            @Override public String getTypeName() { return TypeMemento.Basic.BINARY_BLOB.getTypeName(); }
            @Override public int getFixedSize() { return length; }
            @Override public String toString() { return "BINARY_BLOB(" + length + ")";}
        };
    }


    // --- Series Operations ---
    public static <T> int seriesSize(Series<T> s) { return s == null ? 0 : s.f(); }
    public static <T> T seriesGet(Series<T> s, int i) {
        if (s == null || i < 0 || i >= s.f()) {
             throw new IndexOutOfBoundsException("Index: " + i + ", Size: " + (s == null ? 0 : s.f()));
        }
        return s.s().apply(i);
    }
    public static <T,R> Series<R> seriesMap(Series<T> s, Function<? super T,? extends R> m) {
        Objects.requireNonNull(s, "Source series cannot be null");
        Objects.requireNonNull(m, "Mapping function cannot be null");
        return sr(sz(s), i -> m.apply(get(s,i)));
    }
    public static <T> Series<T> seriesFilter(Series<T> s, Predicate<? super T> p) {
       Objects.requireNonNull(s, "Source series cannot be null");
       Objects.requireNonNull(p, "Predicate cannot be null");
       return new Series<>() {
           private List<Integer> idxCache = null;
           private int sizeCache = -1;

           private synchronized void compute() {
               if (idxCache == null) {
                   idxCache = new ArrayList<>();
                   for(int i=0; i < D.sz(s); i++) {
                       T element = D.get(s,i);
                       if(p.test(element)) {
                           idxCache.add(i);
                       }
                   }
                   sizeCache = idxCache.size();
               }
           }
           @Override public Integer f() { compute(); return sizeCache; }
           @Override public Function<Integer,T> s() {
               compute();
               return i -> {
                   if(i < 0 || i >= sizeCache) throw new IndexOutOfBoundsException("Filtered series index: " + i + ", size: " + sizeCache);
                   return D.get(s, idxCache.get(i));
               };
           }
           @Override public String toString() { return "FilteredSeries(sourceSize=" + D.sz(s) + ", computedSize=" + (sizeCache == -1 ? "lazy" : sizeCache) + ")"; }
       };
    }
    public static <T> Series<T> seriesHead(Series<T> s, int n) {
        Objects.requireNonNull(s, "Source series cannot be null");
        n = Math.max(0, n);
        return sr(Math.min(n, sz(s)), i -> get(s, i));
    }
    public static <T> Series<T> seriesTail(Series<T> s, int n) {
        Objects.requireNonNull(s, "Source series cannot be null");
        n = Math.max(0, n);
        int originalSize = sz(s);
        int newSize = Math.min(n, originalSize);
        return sr(newSize, i -> get(s, originalSize - newSize + i));
    }
    public static <T> Series<T> seriesSkip(Series<T> s, int n) {
        Objects.requireNonNull(s, "Source series cannot be null");
        n = Math.max(0, n);
        int originalSize = sz(s);
        int newSize = Math.max(0, originalSize - n);
        int offset = Math.min(n, originalSize);
        return sr(newSize, i -> get(s, offset + i));
    }
    public static <T> List<T> seriesToList(Series<T> s) {
        Objects.requireNonNull(s, "Source series cannot be null");
        if (sz(s) == 0) return Collections.emptyList();
        return seriesStream(s).collect(Collectors.toList());
    }
    public static <T> Stream<T> seriesStream(Series<T> s) {
        Objects.requireNonNull(s, "Source series cannot be null");
        return IntStream.range(0, sz(s)).mapToObj(i -> get(s,i));
    }
    public static <T> void seriesForEach(Series<T> s, Consumer<? super T> c) {
        Objects.requireNonNull(s, "Source series cannot be null");
        Objects.requireNonNull(c, "Consumer action cannot be null");
        seriesStream(s).forEach(c);
    }
    public static <T> T seriesFirst(Series<T> s) {
        Objects.requireNonNull(s, "Source series cannot be null");
        if (sz(s) == 0) throw new NoSuchElementException("Cannot get first element from an empty series.");
        return get(s,0);
    }
    public static <T> T seriesLast(Series<T> s) {
        Objects.requireNonNull(s, "Source series cannot be null");
        int size = sz(s);
        if (size == 0) throw new NoSuchElementException("Cannot get last element from an empty series.");
        return get(s, size - 1);
    }

    // Terse names for Series Operations
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
    public static Join<Object,Supplier<ColumnMeta>> rowVecGetCell(RowVec rv, int colIdx) { return get(rv, colIdx); }
    public static Object rowVecGetValue(RowVec rv, int colIdx) {
        Join<Object,Supplier<ColumnMeta>> cell = rowVecGetCell(rv, colIdx);
        return cell != null ? cell.f() : null;
    }
    public static String rowVecGetColName(RowVec rv, int colIdx) {
        Join<Object,Supplier<ColumnMeta>> cell = rowVecGetCell(rv, colIdx);
        if (cell == null || cell.s() == null) return null;
        ColumnMeta meta = cell.s().get();
        return meta != null ? meta.f() : null;
    }
    public static TypeMemento rowVecGetColType(RowVec rv, int colIdx) {
        Join<Object,Supplier<ColumnMeta>> cell = rowVecGetCell(rv, colIdx);
        if (cell == null || cell.s() == null) return null;
        ColumnMeta meta = cell.s().get();
        return meta != null ? meta.s() : null;
    }

    // Terse names for RowVec Operations
    public static Join<Object,Supplier<ColumnMeta>> cell(RowVec rv, int colIdx) { return rowVecGetCell(rv,colIdx); }
    public static Object get(RowVec rv, int colIdx) { return rowVecGetValue(rv,colIdx); } // Overloaded
    public static String colName(RowVec rv, int colIdx) { return rowVecGetColName(rv,colIdx); }
    public static TypeMemento colType(RowVec rv, int colIdx) { return rowVecGetColType(rv,colIdx); }

    // New overloaded methods for RowVec by column name
    public static Object get(RowVec rv, String colName) {
        Objects.requireNonNull(rv, "RowVec cannot be null");
        Objects.requireNonNull(colName, "Column name cannot be null");
        for (int i = 0; i < sz(rv); i++) {
            if (colName.equals(colName(rv, i))) {
                return get(rv, i);
            }
        }
        throw new IllegalArgumentException("Column '" + colName + "' not found in RowVec.");
    }

    public static TypeMemento colType(RowVec rv, String colName) {
        Objects.requireNonNull(rv, "RowVec cannot be null");
        Objects.requireNonNull(colName, "Column name cannot be null");
        for (int i = 0; i < sz(rv); i++) {
            if (colName.equals(colName(rv, i))) {
                return colType(rv, i);
            }
        }
        throw new IllegalArgumentException("Column '" + colName + "' not found in RowVec.");
    }


    // --- Cursor Operations ---
    public static RowVec cursorGetRow(Cursor c, int rowIdx) { return get(c, rowIdx); }
    public static Cursor cursorMapRows(Cursor c, Function<RowVec,RowVec> m) {
        Objects.requireNonNull(c, "Source cursor cannot be null");
        Objects.requireNonNull(m, "Mapping function cannot be null");
        return (Cursor) map(c,m);
    }
    public static Cursor cursorFilterRows(Cursor c, Predicate<RowVec> p) {
        Objects.requireNonNull(c, "Source cursor cannot be null");
        Objects.requireNonNull(p, "Predicate cannot be null");
        return (Cursor) flt(c,p);
    }

    // Terse names for Cursor Operations
    public static RowVec get(Cursor c, int rowIdx) { return cursorGetRow(c,rowIdx); } // Overloaded
    public static Cursor mapRow(Cursor c, Function<RowVec,RowVec> m) { return cursorMapRows(c,m); }
    public static Cursor fltRow(Cursor c, Predicate<RowVec> p) { return cursorFilterRows(c,p); }


    // --- ISAM File Handling ---
    public static final String META_SFX = ".meta";
    public static final String DATA_SFX = ".data";
    private static final ByteOrder ISAM_BYTE_ORDER = ByteOrder.BIG_ENDIAN;

    public record IsamFileMetadata(List<ColumnMeta> schema, long recordCount, int recordByteLength) {
        public void write(DataOutput out) throws IOException {
            out.writeInt(schema.size());
            for (ColumnMeta cm : schema) {
                out.writeUTF(cm.f()); // name
                out.writeUTF(cm.s().getTypeName()); // base type name (e.g. "String", "Integer")
                out.writeInt(cm.s().getFixedSize()); // actual fixed size used (e.g. 100 for String(100))
            }
            out.writeLong(recordCount);
            out.writeInt(recordByteLength);
        }

        public static IsamFileMetadata read(DataInput in) throws IOException {
            int schemaSize = in.readInt();
            if (schemaSize < 0) throw new IOException("Invalid schema size in metadata: " + schemaSize);
            List<ColumnMeta> schema = new ArrayList<>(schemaSize);
            int calculatedRecordByteLength = 0;

            for (int i = 0; i < schemaSize; i++) {
                String name = in.readUTF();
                String baseTypeName = in.readUTF();
                int actualFixedSize = in.readInt();

                TypeMemento type;
                if (TypeMemento.Basic.STRING.getTypeName().equals(baseTypeName)) {
                    if (actualFixedSize <= 0) throw new IOException("ISAM String column '" + name + "' has invalid fixed size: " + actualFixedSize);
                    type = D.fsString(actualFixedSize);
                } else if (TypeMemento.Basic.BINARY_BLOB.getTypeName().equals(baseTypeName)) {
                    if (actualFixedSize <= 0) throw new IOException("ISAM BinaryBlob column '" + name + "' has invalid fixed size: " + actualFixedSize);
                    type = D.fsBinaryBlob(actualFixedSize);
                } else {
                    type = TypeMemento.Basic.fromTypeName(baseTypeName);
                    if (type.getFixedSize() != actualFixedSize) {
                         throw new IOException("Mismatch in fixedSize for basic type " + baseTypeName + " column '" + name +
                                              "'. Expected from enum: " + type.getFixedSize() + ", Stored in ISAM: " + actualFixedSize);
                    }
                     if (type.getFixedSize() <= 0) { // Should be caught by above for basic types
                        throw new IOException("Basic type " + baseTypeName + " column '" + name + "' must have a positive fixed size for ISAM, but resolved to " + type.getFixedSize());
                    }
                }
                schema.add(D.cm(name, type));
                calculatedRecordByteLength += actualFixedSize;
            }

            long recordCount = in.readLong();
            int recordByteLength = in.readInt();

            if (recordByteLength != calculatedRecordByteLength && schemaSize > 0) {
                throw new IOException("Mismatch in recordByteLength. Calculated: " + calculatedRecordByteLength + ", Stored: " + recordByteLength);
            }
            return new IsamFileMetadata(Collections.unmodifiableList(schema), recordCount, recordByteLength);
        }
    }

    public static class IsamCursor implements Cursor, AutoCloseable {
        private final String pathBase;
        private final IsamFileMetadata metadata;
        private RandomAccessFile dataFile;

        public IsamCursor(String pathBase) throws IOException {
            this.pathBase = pathBase;
            try (DataInputStream metaIn = new DataInputStream(new BufferedInputStream(new FileInputStream(pathBase + META_SFX)))) {
                this.metadata = IsamFileMetadata.read(metaIn);
            }
            // dataFile is opened lazily or can be opened here
        }

        private RandomAccessFile getDataFile() throws IOException {
            if (dataFile == null) {
                dataFile = new RandomAccessFile(pathBase + DATA_SFX, "r");
            }
            return dataFile;
        }

        @Override
        public Integer f() {
            if (metadata.recordCount() > Integer.MAX_VALUE) {
                System.err.println("Warning: ISAM record count exceeds Integer.MAX_VALUE. Cursor size will be truncated.");
                return Integer.MAX_VALUE;
            }
            return (int) metadata.recordCount();
        }

        @Override
        public Function<Integer, RowVec> s() {
            return rowIndex -> {
                if (rowIndex < 0 || rowIndex >= metadata.recordCount()) {
                    throw new IndexOutOfBoundsException("Row index: " + rowIndex + ", Record count: " + metadata.recordCount());
                }
                try {
                    RandomAccessFile raf = getDataFile();
                    byte[] rowBytes = new byte[metadata.recordByteLength()];
                    raf.seek((long)rowIndex * metadata.recordByteLength());
                    raf.readFully(rowBytes);
                    return new IsamRowVec(ByteBuffer.wrap(rowBytes).order(ISAM_BYTE_ORDER), metadata.schema());
                } catch (IOException e) {
                    throw new UncheckedIOException("Failed to read row " + rowIndex, e);
                }
            };
        }

        @Override
        public void close() throws IOException {
            if (dataFile != null) {
                dataFile.close();
                dataFile = null;
            }
        }
        @Override
        public String toString() { return "IsamCursor(path=" + pathBase + ", records=" + metadata.recordCount() + ")"; }
    }

    private static class IsamRowVec implements RowVec {
        private final ByteBuffer rowBuffer;
        private final List<ColumnMeta> schema;
        private final int[] columnOffsets;


        public IsamRowVec(ByteBuffer rowBuffer, List<ColumnMeta> schema) {
            this.rowBuffer = rowBuffer;
            this.schema = schema;
            this.columnOffsets = new int[schema.size()];
            int currentOffset = 0;
            for (int i = 0; i < schema.size(); i++) {
                this.columnOffsets[i] = currentOffset;
                currentOffset += schema.get(i).s().getFixedSize();
            }
        }

        @Override
        public Integer f() { return schema.size(); }

        @Override
        public Function<Integer, Join<Object, Supplier<ColumnMeta>>> s() {
            return colIndex -> {
                if (colIndex < 0 || colIndex >= schema.size()) {
                    throw new IndexOutOfBoundsException("Column index: " + colIndex + ", Column count: " + schema.size());
                }
                ColumnMeta columnMeta = schema.get(colIndex);
                int offset = columnOffsets[colIndex];

                ByteBuffer cellBuffer = rowBuffer.duplicate().order(ISAM_BYTE_ORDER);
                cellBuffer.position(offset);
                cellBuffer.limit(offset + columnMeta.s().getFixedSize());

                Object value = readValueFromBuffer(columnMeta.s(), cellBuffer);
                return jn(value, () -> columnMeta);
            };
        }
        @Override public String toString() { return "IsamRowVec(cols=" + schema.size() + ")"; }
    }

    public static void writeValueToBuffer(Object value, TypeMemento type, ByteBuffer buffer) {
        int fixedSize = type.getFixedSize();
        if (fixedSize <= 0) {
             throw new IllegalArgumentException("Type " + type.getTypeName() + " requires a positive fixed size for ISAM writing. Got: " + fixedSize);
        }

        String typeName = type.getTypeName();

        if (value == null) {
            // Handle nulls: typically by writing all zeros for the field's fixed size.
            // This requires that readValueFromBuffer can distinguish actual zeros from null padding.
            // For simplicity here, we might assume non-null, or specific types handle nulls.
            // For STRING/BINARY_BLOB, padding is already part of the logic.
            // For numeric types, 0 is a valid value. A separate null indicator (e.g. bitmap) would be more robust.
            // For this example, let's write zeros for primitive types if null.
            if (type instanceof TypeMemento.Basic basicType) {
                 switch (basicType) {
                    case BOOLEAN: buffer.put((byte)0); break;
                    case BYTE: buffer.put((byte)0); break;
                    case SHORT: buffer.putShort((short)0); break;
                    case CHAR: buffer.putChar('\0'); break;
                    case INTEGER: buffer.putInt(0); break;
                    case LONG: buffer.putLong(0L); break;
                    case FLOAT: buffer.putFloat(0.0f); break;
                    case DOUBLE: buffer.putDouble(0.0); break;
                    case STRING: //fallthrough
                    case BINARY_BLOB: // fallthrough
                        buffer.put(new byte[fixedSize]); break;
                    default:  throw new IllegalArgumentException("Cannot write null for ISAM type: " + typeName);
                 }
                 return;
            } else if (type instanceof FixedSizeTypeMemento) { // fsString or fsBinaryBlob
                 buffer.put(new byte[fixedSize]); // Pad with null bytes
                 return;
            }
             throw new IllegalArgumentException("Cannot write null for ISAM type: " + typeName);
        }


        if (TypeMemento.Basic.BOOLEAN.getTypeName().equals(typeName)) {
            buffer.put((byte) ((Boolean) value ? 1 : 0));
        } else if (TypeMemento.Basic.BYTE.getTypeName().equals(typeName)) {
            buffer.put((Byte) value);
        } else if (TypeMemento.Basic.SHORT.getTypeName().equals(typeName)) {
            buffer.putShort((Short) value);
        } else if (TypeMemento.Basic.INTEGER.getTypeName().equals(typeName)) {
            buffer.putInt((Integer) value);
        } else if (TypeMemento.Basic.LONG.getTypeName().equals(typeName)) {
            buffer.putLong((Long) value);
        } else if (TypeMemento.Basic.FLOAT.getTypeName().equals(typeName)) {
            buffer.putFloat((Float) value);
        } else if (TypeMemento.Basic.DOUBLE.getTypeName().equals(typeName)) {
            buffer.putDouble((Double) value);
        } else if (TypeMemento.Basic.CHAR.getTypeName().equals(typeName)) {
            buffer.putChar((Character) value);
        } else if (TypeMemento.Basic.STRING.getTypeName().equals(typeName) && type instanceof FixedSizeTypeMemento) {
            byte[] strBytes = value.toString().getBytes(StandardCharsets.UTF_8);
            if (strBytes.length <= fixedSize) {
                buffer.put(strBytes);
                buffer.put(new byte[fixedSize - strBytes.length]); // Pad
            } else {
                buffer.put(strBytes, 0, fixedSize); // Truncate
            }
        } else if (TypeMemento.Basic.BINARY_BLOB.getTypeName().equals(typeName) && type instanceof FixedSizeTypeMemento) {
            byte[] blobBytes = (byte[]) value;
             if (blobBytes.length <= fixedSize) {
                buffer.put(blobBytes);
                buffer.put(new byte[fixedSize - blobBytes.length]); // Pad
            } else {
                buffer.put(blobBytes, 0, fixedSize); // Truncate
            }
        } else {
            throw new IllegalArgumentException("Unsupported or improperly defined type for ISAM writing: " + type.getTypeName() + " with fixed size " + fixedSize);
        }
    }

    public static Object readValueFromBuffer(TypeMemento type, ByteBuffer buffer) {
        int fixedSize = type.getFixedSize();
        if (fixedSize <= 0) {
            throw new IllegalArgumentException("Type " + type.getTypeName() + " has invalid fixed size " + fixedSize + " for ISAM reading.");
        }
        String typeName = type.getTypeName();

        if (TypeMemento.Basic.BOOLEAN.getTypeName().equals(typeName)) {
            return buffer.get() == 1;
        } else if (TypeMemento.Basic.BYTE.getTypeName().equals(typeName)) {
            return buffer.get();
        } else if (TypeMemento.Basic.SHORT.getTypeName().equals(typeName)) {
            return buffer.getShort();
        } else if (TypeMemento.Basic.INTEGER.getTypeName().equals(typeName)) {
            return buffer.getInt();
        } else if (TypeMemento.Basic.LONG.getTypeName().equals(typeName)) {
            return buffer.getLong();
        } else if (TypeMemento.Basic.FLOAT.getTypeName().equals(typeName)) {
            return buffer.getFloat();
        } else if (TypeMemento.Basic.DOUBLE.getTypeName().equals(typeName)) {
            return buffer.getDouble();
        } else if (TypeMemento.Basic.CHAR.getTypeName().equals(typeName)) {
            return buffer.getChar();
        } else if (TypeMemento.Basic.STRING.getTypeName().equals(typeName) && type instanceof FixedSizeTypeMemento) {
            byte[] readBytes = new byte[fixedSize];
            buffer.get(readBytes);
            int len = 0;
            while (len < fixedSize && readBytes[len] != 0) len++;
            return new String(readBytes, 0, len, StandardCharsets.UTF_8);
        } else if (TypeMemento.Basic.BINARY_BLOB.getTypeName().equals(typeName) && type instanceof FixedSizeTypeMemento) {
            byte[] readBytes = new byte[fixedSize];
            buffer.get(readBytes);
            // Could check for padding and return only actual data if a convention is used
            return readBytes;
        } else {
            throw new IllegalArgumentException("Unsupported or improperly defined type for ISAM reading: " + type.getTypeName());
        }
    }


    // --- CSV Utilities ---
    public static List<String> parseCsvLine(String line, char delimiter, char quote) {
        List<String> fields = new ArrayList<>();
        if (line == null || line.isEmpty()) return fields;

        StringBuilder currentField = new StringBuilder();
        boolean inQuotes = false;

        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (inQuotes) {
                if (c == quote) {
                    if (i + 1 < line.length() && line.charAt(i + 1) == quote) { // ""
                        currentField.append(quote);
                        i++;
                    } else {
                        inQuotes = false;
                    }
                } else {
                    currentField.append(c);
                }
            } else {
                if (c == delimiter) {
                    fields.add(currentField.toString());
                    currentField.setLength(0);
                } else if (c == quote && currentField.length() == 0) {
                    inQuotes = true;
                } else {
                    currentField.append(c);
                }
            }
        }
        fields.add(currentField.toString());
        return fields;
    }

    public static Cursor readCsv(String csvFilePath, boolean hasHeader, char delimiter, char quoteChar, int sampleLinesForTypeDeduction) throws IOException {
        List<String> lines = Files.readAllLines(Paths.get(csvFilePath), StandardCharsets.UTF_8);
        if (lines.isEmpty()) return createCursor(0, i -> null);

        List<ColumnMeta> schema;
        int dataStartIndex = 0;
        int numCols;

        if (hasHeader) {
            if (lines.isEmpty()) throw new IOException("CSV file has no header line.");
            List<String> headerNames = parseCsvLine(lines.get(0), delimiter, quoteChar);
            numCols = headerNames.size();
            schema = headerNames.stream()
                .map(name -> cm(name, TypeMemento.Basic.UNKNOWN)) // Initial unknown type
                .collect(Collectors.toList());
            dataStartIndex = 1;
        } else {
            numCols = parseCsvLine(lines.get(0), delimiter, quoteChar).size();
            if (numCols == 0 && lines.size() > 1) numCols = parseCsvLine(lines.get(1), delimiter, quoteChar).size(); // Try next line if first is empty
            if (numCols == 0) return createCursor(0, i->null); // No columns
            schema = IntStream.range(0, numCols)
                .mapToObj(i -> cm("column_" + i, TypeMemento.Basic.UNKNOWN))
                .collect(Collectors.toList());
        }

        // --- Type Deduction ---
        List<TypeEvidence> evidences = new ArrayList<>(numCols);
        for (int i = 0; i < numCols; i++) evidences.add(new TypeEvidence());

        int linesToSample = Math.min(lines.size() - dataStartIndex, sampleLinesForTypeDeduction);
        if (sampleLinesForTypeDeduction == -1) linesToSample = lines.size() - dataStartIndex; // Sample all

        for (int i = 0; i < linesToSample; i++) {
            String line = lines.get(dataStartIndex + i);
            List<String> fieldValues = parseCsvLine(line, delimiter, quoteChar);
            for (int j = 0; j < Math.min(numCols, fieldValues.size()); j++) {
                evidences.get(j).assess(fieldValues.get(j));
            }
        }

        List<ColumnMeta> deducedSchema = new ArrayList<>(numCols);
        for (int i = 0; i < numCols; i++) {
            TypeMemento deducedType = evidences.get(i).deduceFinalType();
            // If deduced type is STRING, ensure it gets a fixed size from maxStrLength for ISAM compatibility
            if (TypeMemento.Basic.STRING.getTypeName().equals(deducedType.getTypeName())) {
                deducedType = fsString(Math.max(1, evidences.get(i).maxStrLength)); // Use fsString for deduced strings
            }
            deducedSchema.add(cm(schema.get(i).f(), deducedType)); // Use original name, new type
        }
        final List<ColumnMeta> finalSchema = Collections.unmodifiableList(deducedSchema);
        System.out.println("Deduced CSV Schema: " + finalSchema.stream().map(cs -> cs.f() + ":" + cs.s()).collect(Collectors.joining(", ")));


        // --- Create Cursor with Deduced Schema ---
        List<RowVec> rows = new ArrayList<>();
        for (int i = dataStartIndex; i < lines.size(); i++) {
            List<String> fieldValues = parseCsvLine(lines.get(i), delimiter, quoteChar);
            if (fieldValues.size() != finalSchema.size()) {
                 System.err.println("Warning: CSV line " + (i+1) + " has " + fieldValues.size() + " fields, expected " + finalSchema.size() + ". Padding/truncating.");
                 while(fieldValues.size() < finalSchema.size()) fieldValues.add(""); // Pad
                 while(fieldValues.size() > finalSchema.size()) fieldValues.remove(fieldValues.size()-1); // Truncate
            }

            final List<String> finalFieldValues = fieldValues; // For lambda
            RowVec rv = createRowVec(finalSchema.size(), colIdx -> {
                String csvStrValue = finalFieldValues.get(colIdx);
                ColumnMeta cm = finalSchema.get(colIdx);
                // Attempt to convert to the DEDUCED type here
                Object typedValue;
                try {
                     typedValue = convertCsvStringToTypedValue(csvStrValue, cm.s());
                } catch (IllegalArgumentException e) { // Catch specific parsing errors
                    // If conversion fails for a non-string type, treat as null.
                    // If it's a string type, convertCsvStringToTypedValue would have returned the original string.
                    System.err.println("Warning: Could not convert '" + csvStrValue + "' to " + cm.s().getTypeName() + " for col " + cm.f() + " line " + (i+1) + ". Using null. Error: " + e.getMessage());
                    typedValue = null;
                }
                Supplier<ColumnMeta> cmSupplier = () -> cm;
                return jn(typedValue, cmSupplier);
            });
            rows.add(rv);
        }
        return createCursor(rows.size(), rows::get);
    }


    public static void csvToIsam(String csvFilePath, String isamPathBase, List<ColumnMeta> schema,
                                 boolean csvHasHeader, char delimiter, char quoteChar) throws IOException {
        int recordByteLength = 0;
        for (ColumnMeta cm : schema) {
            int fs = cm.s().getFixedSize();
            if (fs <= 0) {
                throw new IllegalArgumentException("ISAM Schema Error: Column '" + cm.f() + "' type " + cm.s().getTypeName() +
                                                   " has non-positive fixed size (" + fs + "). All ISAM columns must use a positive fixed size (e.g. D.fsString(len) or basic fixed types).");
            }
            recordByteLength += fs;
        }

        List<String> csvLines = Files.readAllLines(Paths.get(csvFilePath), StandardCharsets.UTF_8);
        if (csvLines.isEmpty() && csvHasHeader) throw new IOException("CSV file is empty or has only a header.");
        
        int dataStartIndex = csvHasHeader ? 1 : 0;
        if (dataStartIndex >= csvLines.size() && !csvLines.isEmpty()) { // Only header, no data
             System.out.println("CSV has only header, 0 records to write to ISAM.");
        }
        long recordCount = 0;

        try (RandomAccessFile dataFile = new RandomAccessFile(isamPathBase + DATA_SFX, "rw")) {
            dataFile.setLength(0);
            ByteBuffer rowBuffer = ByteBuffer.allocate(recordByteLength).order(ISAM_BYTE_ORDER);

            for (int i = dataStartIndex; i < csvLines.size(); i++) {
                String line = csvLines.get(i);
                if (line.trim().isEmpty()) continue; // Skip empty lines
                List<String> fieldValues = parseCsvLine(line, delimiter, quoteChar);

                if (fieldValues.size() != schema.size()) {
                    System.err.println("Warning: CSV line " + (i + 1) + " field count " + fieldValues.size() +
                                       " != schema size " + schema.size() + ". Skipping. Line: " + line);
                    continue;
                }

                rowBuffer.clear();
                try {
                    for (int colIdx = 0; colIdx < schema.size(); colIdx++) {
                        ColumnMeta cm = schema.get(colIdx);
                        String csvValue = fieldValues.get(colIdx);
                        Object typedValue = convertCsvStringToTypedValue(csvValue, cm.s());
                        writeValueToBuffer(typedValue, cm.s(), rowBuffer);
                    }
                    dataFile.write(rowBuffer.array(), 0, recordByteLength);
                    recordCount++;
                } catch (Exception e) {
                     System.err.println("Error processing CSV line " + (i+1) + " for ISAM: " + line + ". Error: " + e.getMessage() + ". Skipping.");
                }
            }
        }

        IsamFileMetadata metadata = new IsamFileMetadata(schema, recordCount, recordByteLength);
        try (DataOutputStream metaOut = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(isamPathBase + META_SFX)))) {
            metadata.write(metaOut);
        }
        System.out.println("CSV to ISAM conversion complete. Records written: " + recordCount);
    }

    private static Object convertCsvStringToTypedValue(String csvValue, TypeMemento targetType) {
        String trimmedCsvValue = (csvValue == null) ? "" : csvValue.trim();

        String typeName = targetType.getTypeName();

        if (trimmedCsvValue.isEmpty()) {
            if (TypeMemento.Basic.STRING.getTypeName().equals(typeName)) return "";
            if (TypeMemento.Basic.BOOLEAN.getTypeName().equals(typeName)) return false; // Default for empty
            // For numeric, char, binary types, an empty string means no value, so return null.
            // writeValueToBuffer will handle null by writing zeros.
            return null;
        }

        try {
            if (TypeMemento.Basic.STRING.getTypeName().equals(typeName)) {
                return csvValue; // Keep original for string, not trimmed, as spaces might be intentional
            } else if (TypeMemento.Basic.INTEGER.getTypeName().equals(typeName)) {
                return Integer.parseInt(trimmedCsvValue);
            } else if (TypeMemento.Basic.LONG.getTypeName().equals(typeName)) {
                return Long.parseLong(trimmedCsvValue);
            } else if (TypeMemento.Basic.DOUBLE.getTypeName().equals(typeName)) {
                return Double.parseDouble(trimmedCsvValue);
            } else if (TypeMemento.Basic.FLOAT.getTypeName().equals(typeName)) {
                return Float.parseFloat(trimmedCsvValue);
            } else if (TypeMemento.Basic.BOOLEAN.getTypeName().equals(typeName)) {
                return Boolean.parseBoolean(trimmedCsvValue);
            } else if (TypeMemento.Basic.BYTE.getTypeName().equals(typeName)) {
                return Byte.parseByte(trimmedCsvValue);
            } else if (TypeMemento.Basic.SHORT.getTypeName().equals(typeName)) {
                return Short.parseShort(trimmedCsvValue);
            } else if (TypeMemento.Basic.CHAR.getTypeName().equals(typeName)) {
                if (trimmedCsvValue.length() != 1) throw new IllegalArgumentException("CHAR value must be a single character: '" + trimmedCsvValue + "'");
                return trimmedCsvValue.charAt(0);
            } else if (TypeMemento.Basic.BINARY_BLOB.getTypeName().equals(typeName)) {
                try {
                    return Base64.getDecoder().decode(trimmedCsvValue);
                } catch (IllegalArgumentException e) {
                    throw new IllegalArgumentException("Cannot parse '" + trimmedCsvValue + "' as Base64 for BINARY_BLOB.", e);
                }
            } else {
                throw new IllegalArgumentException("Unsupported type for CSV conversion: " + targetType.getTypeName());
            }
        } catch (NumberFormatException e) {
            // For numeric types, if parsing fails, it's an invalid value.
            // For CHAR, if length is not 1, it's an invalid value.
            // For BINARY_BLOB, if not base64, it's invalid.
            // These should throw IllegalArgumentException.
            throw new IllegalArgumentException("Cannot parse '" + trimmedCsvValue + "' as " + targetType.getTypeName(), e);
        }
    }
}
