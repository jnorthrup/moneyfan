package com.vsiwest.bbcursive.aliases;

import com.vsiwest.bbcursive.collections.Either;
import com.vsiwest.bbcursive.collections.Twin;
import com.vsiwest.bbcursive.core.Join;
import com.vsiwest.bbcursive.core.Series;

public interface JsPathElement extends Either<String, Integer> {}
public interface JsPath extends Series<JsPathElement> {}
public interface JsElement extends Join<Twin<Integer>, Series<Integer>> {}
