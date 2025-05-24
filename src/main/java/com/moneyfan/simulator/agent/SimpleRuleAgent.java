package com.moneyfan.simulator.agent;

import com.moneyfan.dsel.D;
import com.moneyfan.dsel.core.Join;
import com.moneyfan.dsel.core.RowVec;
import com.moneyfan.dsel.core.Series;
import com.moneyfan.simulator.SimWallet;
import com.moneyfan.simulator.model.AssetKey;
import com.moneyfan.simulator.model.AssetMutation;
import com.moneyfan.simulator.model.AssetOutput;

public class SimpleRuleAgent implements TradingAgent {
    private final String id;
    private double previousClose = -1.0;
    private double lastPublishedReward = 0.0; // Keep track of own reward

    public SimpleRuleAgent(String id) { this.id = id; }
    @Override public String getId() { return id; }
    @Override public void initialize(SimWallet wallet, AssetKey assetKey) {
        System.out.printf("Agent %s initialized for %s with wallet.\n", id, assetKey.toPairString());
        previousClose = -1.0; // Reset for new asset or initialization
        lastPublishedReward = 0.0;
    }

    // Updated: matches new TradingAgent.decide signature
    @Override
    public AssetOutput decide(AssetKey assetKey, RowVec kline, SimWallet wallet, Series<Join<String, Double>> sharedData) {
        // Example of using sharedData: print it out for observation
        System.out.printf("[%s] Shared Data for this tick: %s\n", id, D.ls(sharedData));

        // Schema: Timestamp(L), Open(D), High(D), Low(D), Close(D), Volume(D)
        // Indices:    0          1        2        3        4         5
        double currentClose = (Double) D.get(kline, 4); // Get Close Price
        double[] output = new double[AssetMutation.OUTPUT_SIZE];
        for(int i=0; i<output.length; i++) output[i] = AssetMutation.ALL_MUTATIONS[i].defaultValue; // Init with defaults

        if (previousClose > 0) {
            if (currentClose > previousClose * 1.001) { // Price increased by > 0.1%
                output[AssetMutation.BUY_ACTION.ordinal()] = 1.0; // Strong buy signal
                output[AssetMutation.HOLD_ACTION.ordinal()] = 0.0;
                output[AssetMutation.AS_MARKET_ORDER.ordinal()] = 1.0; // Market order
                output[AssetMutation.QUANTITY_FRACTION.ordinal()] = 0.1; // Use 10% of quote balance
                System.out.printf("[%s] %s: Close %.2f > PrevClose %.2f -> DECIDE BUY\n", id, assetKey.toPairString(), currentClose, previousClose);
            } else if (currentClose < previousClose * 0.999) { // Price decreased by > 0.1%
                output[AssetMutation.SELL_ACTION.ordinal()] = 1.0; // Strong sell signal
                output[AssetMutation.HOLD_ACTION.ordinal()] = 0.0;
                output[AssetMutation.AS_MARKET_ORDER.ordinal()] = 1.0; // Market order
                output[AssetMutation.QUANTITY_FRACTION.ordinal()] = 0.1; // Sell 10% of base asset
                System.out.printf("[%s] %s: Close %.2f < PrevClose %.2f -> DECIDE SELL\n", id, assetKey.toPairString(), currentClose, previousClose);
            } else {
                 output[AssetMutation.HOLD_ACTION.ordinal()] = 1.0; // Strong hold
                 System.out.printf("[%s] %s: Close %.2f ~ PrevClose %.2f -> DECIDE HOLD\n", id, assetKey.toPairString(), currentClose, previousClose);
            }
        } else {
            System.out.printf("[%s] %s: No previous close for %s, holding.\n", id, assetKey.toPairString(), assetKey.baseAsset());
            output[AssetMutation.HOLD_ACTION.ordinal()] = 1.0; // Hold if no previous data
        }
        previousClose = currentClose;
        return AssetOutput.create(output);
    }

    // New: implements the publishData method
    @Override
    public void publishData(double reward) {
        // In a real scenario, this would send data to the AgentDataHub
        // For SimpleRuleAgent, we just store it internally for demonstration
        lastPublishedReward = reward;
        // System.out.printf("[%s] Published Reward: %.4f\n", id, reward); // Optional logging
    }
}
