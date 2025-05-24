package com.vsiwest.bbcursive.aliases;

import com.vsiwest.bbcursive.collections.Twin;
import com.vsiwest.bbcursive.core.Join;
import com.vsiwest.bbcursive.core.Series;

public interface JsIndex extends Join<Twin<Integer>, Series<Character>> {}
public interface JsContext extends Join<JsElement, Series<Character>> {}
