package com.moneyfan.simulator;

import com.moneyfan.dsel.D;
import com.moneyfan.dsel.core.Cursor;
import com.moneyfan.dsel.core.RowVec; // Represents a single Kline
import com.moneyfan.simulator.model.AssetKey;
import java.util.Objects;

public class MarketDataStream {
    public final AssetKey assetKey;
    public final Cursor klineCursor; // Cursor of RowVec (Klines)
    private int currentIndex = 0;

    public MarketDataStream(AssetKey assetKey, Cursor klineCursor) {
        this.assetKey = Objects.requireNonNull(assetKey);
        this.klineCursor = Objects.requireNonNull(klineCursor);
    }
    public boolean hasNext() { return currentIndex < D.sz(klineCursor); }
    public RowVec nextKline() { return hasNext() ? D.get(klineCursor, currentIndex++) : null; }
    public RowVec peekNextKline() { return hasNext() ? D.get(klineCursor, currentIndex) : null; }
    public void reset() { currentIndex = 0; }
    public int getStreamSize() { return D.sz(klineCursor); }
}
