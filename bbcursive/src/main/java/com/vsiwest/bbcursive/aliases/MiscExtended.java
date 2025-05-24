package com.vsiwest.bbcursive.aliases;

import com.vsiwest.bbcursive.core.Join;
import com.vsiwest.bbcursive.core.Series;
import com.vsiwest.bbcursive.util.NUID;
import java.util.function.Supplier;

public interface PosixStat extends Supplier<Object> {}
public interface Route<TNum> extends Join<NUID<TNum>, Address> {}
public interface Vect0r<T> extends Series<T> {}
public interface MarketArgTuple extends Supplier<String[]> {}
