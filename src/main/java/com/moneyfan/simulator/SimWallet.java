package com.moneyfan.simulator;

import com.moneyfan.dsel.D;
import com.moneyfan.dsel.core.Join;
import com.moneyfan.simulator.model.AssetKey;
import com.moneyfan.simulator.model.SimOrder;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

public class SimWallet {
    private final Map<AssetKey, Double> baseBalances;
    private final Map<AssetKey, Double> quoteBalances;
    private final String agentId;

    public SimWallet(String agentId) {
        this.agentId = Objects.requireNonNull(agentId);
        this.baseBalances = new HashMap<>();
        this.quoteBalances = new HashMap<>();
    }

    public void initializeBalance(AssetKey assetKey, double baseAmount, double quoteAmount) {
        baseBalances.put(assetKey, baseAmount);
        quoteBalances.put(assetKey, quoteAmount);
    }

    public double getBaseBalance(AssetKey assetKey) { return baseBalances.getOrDefault(assetKey, 0.0); }
    public double getQuoteBalance(AssetKey assetKey) { return quoteBalances.getOrDefault(assetKey, 0.0); }

    public boolean applyTrade(SimOrder.OrderSide side, AssetKey assetKey, double quantity, double price) {
        double baseBalance = getBaseBalance(assetKey);
        double quoteBalance = getQuoteBalance(assetKey);

        if (side == SimOrder.OrderSide.BUY) {
            double cost = quantity * price;
            if (quoteBalance >= cost) {
                baseBalances.put(assetKey, baseBalance + quantity);
                quoteBalances.put(assetKey, quoteBalance - cost);
                System.out.printf("[%s] Wallet: BUY %.4f %s @ %.2f. Cost: %.2f %s. New Base: %.4f, New Quote: %.2f\n",
                    agentId, quantity, assetKey.baseAsset(), price, cost, assetKey.quoteAsset(), getBaseBalance(assetKey), getQuoteBalance(assetKey));
                return true;
            } else {
                System.out.printf("[%s] Wallet: INSUFFICIENT FUNDS to BUY %.4f %s. Need: %.2f %s, Have: %.2f %s\n",
                    agentId, quantity, assetKey.baseAsset(), cost, assetKey.quoteAsset(), quoteBalance, assetKey.quoteAsset());
                return false;
            }
        } else {
            if (baseBalance >= quantity) {
                double proceeds = quantity * price;
                baseBalances.put(assetKey, baseBalance - quantity);
                quoteBalances.put(assetKey, quoteBalance + proceeds);
                System.out.printf("[%s] Wallet: SELL %.4f %s @ %.2f. Proceeds: %.2f %s. New Base: %.4f, New Quote: %.2f\n",
                    agentId, quantity, assetKey.baseAsset(), price, proceeds, assetKey.quoteAsset(), getBaseBalance(assetKey), getQuoteBalance(assetKey));
                return true;
            } else {
                System.out.printf("[%s] Wallet: INSUFFICIENT FUNDS to SELL %.4f %s. Have: %.4f %s\n",
                    agentId, quantity, assetKey.baseAsset(), baseBalance, assetKey.baseAsset());
                return false;
            }
        }
    }
    public double getTotalValue(AssetKey referenceQuoteAsset, Map<AssetKey, Double> currentPrices) {
        double totalValue = 0;
        for(Map.Entry<AssetKey, Double> entry : quoteBalances.entrySet()){
            if(entry.getKey().quoteAsset().equals(referenceQuoteAsset.baseAsset())) {
                totalValue += entry.getValue();
            } else {
                AssetKey conversionPair = AssetKey.of(entry.getKey().quoteAsset() + "/" + referenceQuoteAsset.baseAsset());
                totalValue += entry.getValue() * currentPrices.getOrDefault(conversionPair, 0.0);
            }
        }
        for(Map.Entry<AssetKey, Double> entry : baseBalances.entrySet()){
            AssetKey pair = entry.getKey();
            AssetKey conversionPair = AssetKey.of(pair.baseAsset() + "/" + referenceQuoteAsset.baseAsset());
            totalValue += entry.getValue() * currentPrices.getOrDefault(conversionPair, 0.0);
        }
        return totalValue;
    }
    @Override public String toString() { return "SimWallet{agentId=" + agentId + ", base=" + baseBalances + ", quote=" + quoteBalances + "}"; }
}
