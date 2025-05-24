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

    // Private constructor for creating new immutable instances
    private SimWallet(String agentId, Map<AssetKey, Double> baseBalances, Map<AssetKey, Double> quoteBalances) {
        this.agentId = agentId;
        this.baseBalances = new HashMap<>(baseBalances); // Ensure deep copy for immutability
        this.quoteBalances = new HashMap<>(quoteBalances); // Ensure deep copy for immutability
    }

    // Returns a new SimWallet with updated balance
    public SimWallet initializeBalance(AssetKey assetKey, double baseAmount, double quoteAmount) {
        Map<AssetKey, Double> newBaseBalances = new HashMap<>(baseBalances);
        Map<AssetKey, Double> newQuoteBalances = new HashMap<>(quoteBalances);
        newBaseBalances.put(assetKey, baseAmount);
        newQuoteBalances.put(assetKey, quoteAmount);
        return new SimWallet(agentId, newBaseBalances, newQuoteBalances);
    }

    public double getBaseBalance(AssetKey assetKey) { return baseBalances.getOrDefault(assetKey, 0.0); }
    public double getQuoteBalance(AssetKey assetKey) { return quoteBalances.getOrDefault(assetKey, 0.0); }

    // Returns a Join of (success_boolean, new_wallet_instance)
    public Join<Boolean, SimWallet> applyTrade(SimOrder.OrderSide side, AssetKey assetKey, double quantity, double price) {
        double baseBalance = getBaseBalance(assetKey);
        double quoteBalance = getQuoteBalance(assetKey);

        if (side == SimOrder.OrderSide.BUY) {
            double cost = quantity * price;
            if (quoteBalance >= cost) {
                Map<AssetKey, Double> newBaseBalances = new HashMap<>(baseBalances);
                Map<AssetKey, Double> newQuoteBalances = new HashMap<>(quoteBalances);
                newBaseBalances.put(assetKey, baseBalance + quantity);
                newQuoteBalances.put(assetKey, quoteBalance - cost);
                SimWallet newWallet = new SimWallet(agentId, newBaseBalances, newQuoteBalances);
                System.out.printf("[%s] Wallet: BUY %.4f %s @ %.2f. Cost: %.2f %s. New Base: %.4f, New Quote: %.2f\n",
                    agentId, quantity, assetKey.baseAsset(), price, cost, assetKey.quoteAsset(), newWallet.getBaseBalance(assetKey), newWallet.getQuoteBalance(assetKey));
                return D.jn(true, newWallet);
            } else {
                System.out.printf("[%s] Wallet: INSUFFICIENT FUNDS to BUY %.4f %s. Need: %.2f %s, Have: %.2f %s\n",
                    agentId, quantity, assetKey.baseAsset(), cost, assetKey.quoteAsset(), quoteBalance, assetKey.quoteAsset());
                return D.jn(false, this); // Return current wallet if failed
            }
        } else { // SELL
            if (baseBalance >= quantity) {
                double proceeds = quantity * price;
                Map<AssetKey, Double> newBaseBalances = new HashMap<>(baseBalances);
                Map<AssetKey, Double> newQuoteBalances = new HashMap<>(quoteBalances);
                newBaseBalances.put(assetKey, baseBalance - quantity);
                newQuoteBalances.put(assetKey, quoteBalance + proceeds);
                SimWallet newWallet = new SimWallet(agentId, newBaseBalances, newQuoteBalances);
                System.out.printf("[%s] Wallet: SELL %.4f %s @ %.2f. Proceeds: %.2f %s. New Base: %.4f, New Quote: %.2f\n",
                    agentId, quantity, assetKey.baseAsset(), price, proceeds, assetKey.quoteAsset(), newWallet.getBaseBalance(assetKey), newWallet.getQuoteBalance(assetKey));
                return D.jn(true, newWallet);
            } else {
                System.out.printf("[%s] Wallet: INSUFFICIENT FUNDS to SELL %.4f %s. Have: %.4f %s\n",
                    agentId, quantity, assetKey.baseAsset(), baseBalance, assetKey.baseAsset());
                return D.jn(false, this); // Return current wallet if failed
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
