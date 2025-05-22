package moneyfan.core;

import java.util.function.Supplier;

public record CellMeta(Supplier<Scalar> provider) {}