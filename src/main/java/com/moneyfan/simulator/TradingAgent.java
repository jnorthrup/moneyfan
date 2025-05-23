package com.moneyfan.simulator;

import com.moneyfan.data.Kline;
import com.moneyfan.data.Order;

import java.util.Map;
import java.util.Optional;

public interface TradingAgent {
    String getAgentName();

    void initialize(Wallet initialWallet, Map<String, String> config);

    Optional<Order> decide(Kline currentKline, Wallet currentWallet, Map<String, Object> marketView);
}
