package com.moneyfan.simulator.agent;

import com.moneyfan.dsel.core.RowVec; // Represents a single Kline
import com.moneyfan.simulator.SimWallet;
import com.moneyfan.simulator.model.AssetKey;
import com.moneyfan.simulator.model.AssetOutput;

public interface TradingAgent {
    String getId();
    void initialize(SimWallet wallet, AssetKey assetKey); // Agent might focus on one asset or be general
    AssetOutput decide(AssetKey assetKey, RowVec currentKline, SimWallet wallet); // Kline is RowVec
}
