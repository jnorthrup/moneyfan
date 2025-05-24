package com.vsiwest.bbcursive.core;

import com.vsiwest.bbcursive.aliases.*;
import com.vsiwest.bbcursive.collections.*;
import com.vsiwest.bbcursive.types.ColumnMeta;
import com.vsiwest.bbcursive.types.ColumnMetaImpl;
import com.vsiwest.bbcursive.types.IOMemento;
import com.vsiwest.bbcursive.types.TypeMemento;
import com.vsiwest.bbcursive.util.NUID;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import java.util.function.IntFunction;
import java.util.function.Supplier;
import java.util.List;
import java.util.function.Function;

public enum D { INSTANCE;
    public static <F,S> @NotNull Join<F,S> jn(@Nullable F f, @Nullable S s) { return new Join<>(f,s); }
    public static <T> @NotNull Series<T> sr(int size, @NotNull IntFunction<T> provider) { return Series.SeriesImpl.of(size, provider); }
    public static <T> @NotNull Series<T> sr(@NotNull List<T> list) { return Series.SeriesImpl.of(list.size(), list::get); }
    public static <T> @NotNull Twin<T> tw(@Nullable T f, @Nullable T s) { return new TwinImpl<>(f,s); }
    public static <L,R> @NotNull Either<L,R> left(@Nullable L val) { return new Left<>(val); }
    public static <L,R> @NotNull Either<L,R> right(@Nullable R val) { return new Right<>(val); }
    public static @NotNull ColumnMeta cm(@NotNull String name, @NotNull TypeMemento type) { return new ColumnMetaImpl(name,type); }
    public static @NotNull RowVec rv(int size, @NotNull IntFunction<Join<Object, Supplier<ColumnMeta>>> p) { return (RowVec) Series.SeriesImpl.of(size, p); }
    public static @NotNull RowVec rv(@NotNull List<Join<Object, Supplier<ColumnMeta>>> valuesAndMeta) { return (RowVec) Series.SeriesImpl.of(valuesAndMeta.size(), valuesAndMeta::get); }
    public static @NotNull Cursor cur(int size, @NotNull IntFunction<RowVec> p) { return (Cursor) Series.SeriesImpl.of(size, p); }
    public static @NotNull Cursor cur(@NotNull List<RowVec> rows) { return (Cursor) Series.SeriesImpl.of(rows.size(), rows::get); }

    public static @NotNull Address adr(@NotNull String val) { return () -> val; }
    public static @NotNull AgentAction agentAct(@NotNull double[] val) { return () -> val; }
    public static @NotNull AsyncReaction asyncR(@NotNull Integer i, @NotNull UnaryAsyncReaction uar) { return jn(i, uar); }
    public static @NotNull DelimitRange delimitR(@NotNull Integer begin, @NotNull Integer end) { return tw(begin, end); }
    public static @NotNull Interest interest(@NotNull Integer val) { return () -> val; }
    public static <T> @NotNull LongSeries<T> longS(long size, @NotNull Function<Long, T> p) { return jn(size, p); }
    public static @NotNull MarketArgTuple marketArgT(@NotNull String[] val) { return () -> val; }
    public static @NotNull PosixOffset posixOff(@NotNull Long val) { return () -> val; }
    public static @NotNull PosixStat posixSt(@NotNull Object val) { return () -> val; }
    public static <TNum> @NotNull Route<TNum> route(@NotNull NUID<TNum> nuid, @NotNull Address addr) { return jn(nuid, addr); }
}
