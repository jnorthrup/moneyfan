package com.vsiwest.bbcursive.aliases;

import com.vsiwest.bbcursive.collections.RowVec;
import com.vsiwest.bbcursive.core.Series;
import java.util.function.Supplier;

public interface AgentAction extends Supplier<double[]> {}
public interface AgentObservation extends Series<RowVec> {}
