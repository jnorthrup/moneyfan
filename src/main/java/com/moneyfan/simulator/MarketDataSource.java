package com.moneyfan.simulator;

import com.moneyfan.data.Kline;

public interface MarketDataSource {
    boolean hasNext();
    Kline getNextKline();
}
