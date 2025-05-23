package com.moneyfan.agent;

import com.moneyfan.data.Kline;
import com.moneyfan.data.Order;
import com.moneyfan.data.Wallet;

import java.util.Optional;

public interface TradingAgent {
    Optional<Order> onCandleSignal(Kline kline, Wallet wallet);
}
