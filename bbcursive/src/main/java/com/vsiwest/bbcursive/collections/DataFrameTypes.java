package com.vsiwest.bbcursive.collections;

import com.vsiwest.bbcursive.core.Join;
import com.vsiwest.bbcursive.core.Series;
import com.vsiwest.bbcursive.types.ColumnMeta;
import java.util.function.Supplier;

public interface Series2<A, B> extends Series<Join<A, B>> {}
public interface RowVec extends Series2<Object, Supplier<ColumnMeta>> {}
public interface Cursor extends Series<RowVec> {}
