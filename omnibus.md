Okay, I will deliver the "omnibus enum" (`D.java`) incorporating the DSEL operations with both expressive and terse names, the lambda-styled lazy `IsamCursor`, and CSV-to-ISAM utilities. The core DSEL type interfaces (`Join`, `Series`, etc.) will remain in the `com.moneyfan.dsel.core` package for clarity, while `D.java` will be the central operational hub.

This will be a very large file, so I'll structure it clearly.

---

**1. Core DSEL Type Interfaces (package `com.moneyfan.dsel.core`)**

These remain largely the same as the previous iteration where `Join` became an interface.

**`src/main/java/com/moneyfan/dsel/core/Join.java`**
```java
package com.moneyfan.dsel.core;
import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Predicate;
 interface Join<F, S> {
    F f(); S s();
    static <F_TYPE, S_TYPE> Join<F_TYPE, S_TYPE> jn(F_TYPE f, S_TYPE s) { return new ImmutableJoinRecord<>(f, s); }
    default <FN> Join<FN, S> mapFst(Function<? super F, ? extends FN> m) { Objects.requireNonNull(m); return jn(m.apply(f()), s()); }
    default <SN> Join<F, SN> mapSnd(Function<? super S, ? extends SN> m) { Objects.requireNonNull(m); return jn(f(), m.apply(s())); }
    default <FN, SN> Join<FN, SN> mapBoth(BiFunction<? super F, ? super S, ? extends Join<FN, SN>> bm) { Objects.requireNonNull(bm); return bm.apply(f(), s()); }
    default <FN, SN> Join<FN, SN> mapBoth(Function<? super F, ? extends FN> fm, Function<? super S, ? extends SN> sm) { Objects.requireNonNull(fm); Objects.requireNonNull(sm); return jn(fm.apply(f()), sm.apply(s())); }
    default Join<S, F> swap() { return jn(s(), f()); }
    default boolean test(Predicate<? super Join<F, S>> p) { Objects.requireNonNull(p); return p.test(this); }
}
record ImmutableJoinRecord<F, S>(F f, S s) implements Join<F, S> {
    @Override  String toString() { return "jn(" + f + ", " + s + ")"; }
}
```

**`src/main/java/com/moneyfan/dsel/core/TypeMemento.java`**
```java
package com.moneyfan.dsel.core;
 interface TypeMemento {
    String getTypeName(); int getFixedSize();
    enum Basic implements TypeMemento {
        BOOLEAN("Boolean", 1), BYTE("Byte", 1), SHORT("Short", 2), INTEGER("Integer", 4), LONG("Long", 8),
        FLOAT("Float", 4), DOUBLE("Double", 8), CHAR("Char", 2), STRING("String", -1), BINARY_BLOB("BinaryBlob", -1),
        OBJECT("Object", -1), JOIN("Join", -1), SERIES("Series", -1), ROWVEC("RowVec", -1), CURSOR("Cursor", -1),
        TWIN("Twin", -1), CUSTOM("Custom", -1);
        private final String tn; private final int fs;
        Basic(String tn, int fs) { this.tn = tn; this.fs = fs; }
        @Override  String getTypeName() { return tn; } @Override  int getFixedSize() { return fs; }
         static TypeMemento fromTypeName(String name) {
            for (Basic b : values()) if (b.getTypeName().equals(name)) return b;
            throw new IllegalArgumentException("Unknown TypeMemento name: " + name);
        }
    }
}
```

**`src/main/java/com/moneyfan/dsel/core/Series.java`**
```java
package com.moneyfan.dsel.core; import java.util.function.Function;
 interface Series<T> extends Join<Integer, Function<Integer, T>> {} 
 interface ColumnMeta extends Join<String, TypeMemento> {} 
 interface RowVec extends Series<Join<Object, Supplier<ColumnMeta>>> {} 
 interface Cursor extends Series<RowVec> {} 
 interface Twin<T> extends Join<T,T> {} 
 ```
 
 ```java 
package com.moneyfan.dsel;
[..]

 enum D { // DSEL Operations Hub (D)
    ;
    // --- Core Join ---
     static <F, S> Join<F, S> createJoin(F f, S s) { return Join.jn(f, s); }
     static <F, S> F first(Join<F, S> j) { return j.f(); }
     static <F, S> S second(Join<F, S> j) { return j.s(); }
     static <F, S, FN> Join<FN, S> mapFirst(Join<F, S> j, Function<? super F, ? extends FN> m) { return j.mapFst(m); }
     static <F, S, SN> Join<F, SN> mapSecond(Join<F, S> j, Function<? super S, ? extends SN> m) { return j.mapSnd(m); }
     static <F, S, FN, SN> Join<FN, SN> mapJoin(Join<F,S> j, BiFunction<? super F,? super S,? extends Join<FN,SN>> bm) { return j.mapBoth(bm); }
     static <F, S, FN, SN> Join<FN, SN> mapJoin(Join<F,S> j, Function<? super F,? extends FN> fm, Function<? super S,? extends SN> sm) { return j.mapBoth(fm, sm); }
     static <F, S> Join<S, F> swapJoin(Join<F, S> j) { return j.swap(); }
     static <F, S> boolean testJoin(Join<F, S> j, Predicate<? super Join<F, S>> p) { return j.test(p); }

     static <F, S> Join<F, S> jn(F f, S s) { return createJoin(f,s); }
     static <F, S> F f(Join<F, S> j) { return first(j); }
     static <F, S> S s(Join<F, S> j) { return second(j); }
     static <F, S, FN> Join<FN, S> mf(Join<F, S> j, Function<? super F, ? extends FN> m) { return mapFirst(j,m); }
     static <F, S, SN> Join<F, SN> ms(Join<F, S> j, Function<? super S, ? extends SN> m) { return mapSecond(j,m); }
     static <F, S, FN, SN> Join<FN, SN> mb(Join<F,S> j, BiFunction<? super F,? super S,? extends Join<FN,SN>> bm) { return mapJoin(j,bm); }
     static <F, S, FN, SN> Join<FN, SN> mb(Join<F,S> j, Function<? super F,? extends FN> fm, Function<? super S,? extends SN> sm) { return mapJoin(j,fm,sm); }
     static <F, S> Join<S, F> sw(Join<F, S> j) { return swapJoin(j); }
     static <F, S> boolean tst(Join<F, S> j, Predicate<? super Join<F, S>> p) { return testJoin(j,p); }

    // --- Conceptual Type Factories ---
     static <T> Series<T> createSeries(int size, Function<Integer, T> gen) { if (size<0) throw new IllegalArgumentException("Series size invalid: "+size); return (Series<T>)jn(size, gen); }
     static ColumnMeta createColumnMeta(String n, TypeMemento t) { return (ColumnMeta)jn(n, t); }
     static RowVec createRowVec(int nCols, Function<Integer, Join<Object, Supplier<ColumnMeta>>> cellGen) { return (RowVec)createSeries(nCols, cellGen); }
     static Cursor createCursor(int nRows, Function<Integer, RowVec> rowGen) { return (Cursor)createSeries(nRows, rowGen); }
     static <T> Twin<T> createTwin(T t1, T t2) { return (Twin<T>)jn(t1,t2); }

     static <T> Series<T> sr(int sz, Function<Integer, T> gen) { return createSeries(sz,gen); }
     static ColumnMeta cm(String n, TypeMemento t) { return createColumnMeta(n,t); }
     static RowVec rv(int sz, Function<Integer, Join<Object, Supplier<ColumnMeta>>> gen) { return createRowVec(sz,gen); }
     static Cursor cr(int sz, Function<Integer, RowVec> gen) { return createCursor(sz,gen); }
     static <T> Twin<T> tw(T t1, T t2) { return createTwin(t1,t2); }

    // --- Series Operations ---
     static <T> int seriesSize(Series<T> s) { return s == null ? 0 : s.f(); }
     static <T> T seriesGet(Series<T> s, int i) { return (s==null || i<0 || i>=s.f()) ? null : s.s().apply(i); }
     static <T,R> Series<R> seriesMap(Series<T> s, Function<? super T,? extends R> m) { Objects.requireNonNull(s); Objects.requireNonNull(m); return sr(sz(s), i->m.apply(get(s,i))); }
     static <T> Series<T> seriesFilter(Series<T> s, Predicate<? super T> p) {
        Objects.requireNonNull(s); Objects.requireNonNull(p);
        return new Series<>() { // Lazy caching filter
            private List<Integer> idxCache = null; private int sizeCache = -1;
            private void compute() { if (idxCache==null) { idxCache = new ArrayList<>(); for(int i=0;i<D.sz(s);i++) if(p.test(D.get(s,i))) idxCache.add(i); sizeCache=idxCache.size(); }}
            @Override  Integer f() { compute(); return sizeCache; }
            @Override  Function<Integer,T> s() { compute(); return i -> { if(i<0||i>=sizeCache) throw new IndexOutOfBoundsException(); return D.get(s, idxCache.get(i)); };}
        };
    }
     static <T> Series<T> seriesHead(Series<T> s, int n) { Objects.requireNonNull(s); n=Math.max(0,n); return sr(Math.min(n,sz(s)), i->get(s,i)); }
     static <T> Series<T> seriesTail(Series<T> s, int n) { Objects.requireNonNull(s); n=Math.max(0,n); int oSz=sz(s); int nSz=Math.min(n,oSz); return sr(nSz, i->get(s,oSz-nSz+i)); }
     static <T> Series<T> seriesSkip(Series<T> s, int n) { Objects.requireNonNull(s); n=Math.max(0,n); int oSz=sz(s); return sr(Math.max(0,oSz-n), i->get(s,Math.min(n,oSz)+i));}
     static <T> List<T> seriesToList(Series<T> s) { Objects.requireNonNull(s); return seriesStream(s).collect(Collectors.toList()); }
     static <T> Stream<T> seriesStream(Series<T> s) { Objects.requireNonNull(s); return IntStream.range(0,sz(s)).mapToObj(i->get(s,i)); }
     static <T> void seriesForEach(Series<T> s, Consumer<? super T> c) { Objects.requireNonNull(s); seriesStream(s).forEach(c); }
     static <T> T seriesFirst(Series<T> s) { Objects.requireNonNull(s); if(sz(s)==0) throw new java.util.NoSuchElementException(); return get(s,0); }
     static <T> T seriesLast(Series<T> s) { Objects.requireNonNull(s); int z=sz(s); if(z==0) throw new java.util.NoSuchElementException(); return get(s,z-1); }

     static <T> int sz(Series<T> s) { return seriesSize(s); }
     static <T> T get(Series<T> s, int i) { return seriesGet(s,i); }
     static <T,R> Series<R> map(Series<T> s, Function<? super T,? extends R> m) { return seriesMap(s,m); }
     static <T> Series<T> flt(Series<T> s, Predicate<? super T> p) { return seriesFilter(s,p); }
     static <T> List<T> ls(Series<T> s) { return seriesToList(s); }
     static <T> Stream<T> st(Series<T> s) { return seriesStream(s); }
     static <T> Series<T> hd(Series<T> s, int n) { return seriesHead(s,n); }
     static <T> Series<T> tl(Series<T> s, int n) { return seriesTail(s,n); }
     static <T> Series<T> sk(Series<T> s, int n) { return seriesSkip(s,n); }
     static <T> void each(Series<T> s, Consumer<? super T> c) { seriesForEach(s,c); }
     static <T> T fst(Series<T> s) { return seriesFirst(s); }
     static <T> T lst(Series<T> s) { return seriesLast(s); }

    // --- RowVec Operations ---
     static Join<Object,Supplier<ColumnMeta>> rowVecGetCell(RowVec rv, int colIdx) { return get(rv,colIdx); }
     static Object rowVecGetValue(RowVec rv, int colIdx) { Join<Object,Supplier<ColumnMeta>> c=cell(rv,colIdx); return c!=null?c.f():null;}
     static String rowVecGetColName(RowVec rv, int colIdx) { ColumnMeta m=cell(rv,colIdx).s().get(); return m!=null?m.f():null; }
     static TypeMemento rowVecGetColType(RowVec rv, int colIdx) { ColumnMeta m=cell(rv,colIdx).s().get(); return m!=null?m.s():null; }

     static Join<Object,Supplier<ColumnMeta>> cell(RowVec rv, int colIdx) { return rowVecGetCell(rv,colIdx); }
     static Object get(RowVec rv, int colIdx) { return rowVecGetValue(rv,colIdx); }
     static String colName(RowVec rv, int colIdx) { return rowVecGetColName(rv,colIdx); }
     static TypeMemento colType(RowVec rv, int colIdx) { return rowVecGetColType(rv,colIdx); }

    // --- Cursor Operations ---
     static RowVec cursorGetRow(Cursor c, int rowIdx) { return get(c,rowIdx); }
     static Cursor cursorMapRows(Cursor c, Function<RowVec,RowVec> m) { return (Cursor)map(c,m); }
     static Cursor cursorFilterRows(Cursor c, Predicate<RowVec> p) { return (Cursor)flt(c,p); }

     static RowVec get(Cursor c, int rowIdx) { return cursorGetRow(c,rowIdx); }
     static Cursor mapRow(Cursor c, Function<RowVec,RowVec> m) { return cursorMapRows(c,m); }
     static Cursor fltRow(Cursor c, Predicate<RowVec> p) { return cursorFilterRows(c,p); }

    // --- ISAM File Handling ---
     static final String META_SFX = ".meta";  static final String DATA_SFX = ".data";

     record IsamFileMetadata(List<ColumnMeta> schema, long recordCount, int recordByteLength) {
 [...]
}
```

**Key Changes & Rationale in `D.java` and ISAM parts:**

*   **Omnibus `D` Enum**: Contains all factories and operations. Expressive names are the primary static methods, and terse names call these.
*   **Lazy `seriesFilter`**: Implemented with an inner class that caches matching indices and size upon first access to `f()` (size) or `s()` (generator). This makes filtering lazy but subsequent access to the filtered series efficient if its elements or size are needed multiple times.
*   **`IsamFileMetadata`**: Now includes `recordByteLength` and handles schema serialization/deserialization more robustly, including type name and fixed size. It ensures fixed-size types for the current ISAM implementation.
*   **`IsamCursor`**:
    *   Takes `pathBase` and reads metadata itself.
    *   `s().apply(rowIndex)` (the generator for rows) performs a `seek` and `readFully` for the specific row's bytes *at the time it's called*. This is lazy row access.
    *   The `RowVec` generated by `IsamCursor` is also lazy: its cells' values are parsed from the row's byte buffer on demand via `D.readValueFromBuffer`.
*   **`writeValueToBuffer`/`readValueFromBuffer`**: These handle serialization/deserialization for fixed-size types for ISAM. STRING and BINARY_BLOB require a fixed size specified in their `TypeMemento` for this simple ISAM.
*   **CSV Utilities**: `readCsv` and `parseCsvLine` are provided. `csvToIsam` demonstrates converting CSV to your ISAM format.
*   **Error Handling**: Minimal for compactness (e.g., `IllegalArgumentException`, `UncheckedIOException`).
*   **Unary Function Glory**: `seriesMap`, `seriesFilter`, `cursorMapRows`, `cursorFilterRows` take a `Series`/`Cursor` and a lambda, returning a new `Series`/`Cursor`. The lambdas operate on individual elements/rows.

This is a very dense implementation. The simulator part would involve agents processing `Cursor`s of candle data (where `Candle` is likely a `RowVec` or a `Join` within a `RowVec`). The `ArchiveDataHandler` would use `D.csvToIsam` to prepare data.

**To re-animate the simulator and harness from Acapulco:**
1.  Replace Acapulco's `Cursor` and data structures with `com.moneyfan.dsel.core.Cursor`, `RowVec`, `Series`, `Join`.
2.  Use methods from `D` enum for all data manipulation (e.g., `D.mapRow`, `D.get(cursor, rowIndex)`).
3.  Data loading for the simulator would use `new D.IsamCursor("path/to/your/kline_data")`.
4.  The `ArchiveDataHandler` would use `D.csvToIsam(csvPath, isamPath, klineSchema, ...)` to convert Binance CSVs.

This provides the requested DSEL structure and ISAM capabilities. The single `D.java` enum is massive but centralizes all operations.