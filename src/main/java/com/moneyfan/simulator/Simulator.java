package com.moneyfan.simulator;

import com.moneyfan.dsel.D;
import com.moneyfan.dsel.core.RowVec;
import com.moneyfan.dsel.core.Join;
import com.moneyfan.dsel.core.Series;
import com.moneyfan.simulator.agent.TradingAgent;
import com.moneyfan.simulator.model.AssetKey;
import com.moneyfan.simulator.model.AssetMutation;
import com.moneyfan.simulator.model.AssetOutput;
import com.moneyfan.simulator.model.SimOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class Simulator {
    private final Map<AssetKey, MarketDataStream> marketDataStreams = new TreeMap<>();
    private final List<Join<TradingAgent, AssetKey>> agentAssignments = new ArrayList<>();
    private Map<String, SimWallet> agentWallets = new HashMap<>();
    private long currentTick = 0;
    private final AssetKey referenceFiat = AssetKey.of("USDT/USDT");
    private final ExecutorService executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
    private final AgentDataHub dataHub = new AgentDataHub();

    public void addMarketData(MarketDataStream stream) {
        marketDataStreams.put(stream.assetKey, stream);
    }

    public void registerAgent(TradingAgent agent, AssetKey assetToTrade, double initialBase, double initialQuote) {
        SimWallet wallet = agentWallets.computeIfAbsent(agent.getId(), SimWallet::new);
        SimWallet updatedWallet = wallet.initializeBalance(assetToTrade, initialBase, initialQuote);
        agentWallets.put(agent.getId(), updatedWallet);
        agent.initialize(updatedWallet, assetToTrade);
        agentAssignments.add(D.jn(agent, assetToTrade));
        dataHub.registerAgent(agent.getId());
    }

    public void runSimulation(long totalTicks) {
        System.out.println("Starting simulation for " + totalTicks + " ticks.");
        for (currentTick = 0; currentTick < totalTicks; currentTick++) {
            if (marketDataStreams.values().stream().noneMatch(MarketDataStream::hasNext)) {
                System.out.println("All market data streams exhausted at tick " + currentTick);
                break;
            }
            System.out.printf("\n--- Tick %d ---\n", currentTick);
            processTick();
            if (currentTick > 0 && currentTick % 100 == 0) printPortfolioValues();
        }
        System.out.println("\nSimulation finished at tick " + currentTick);
        printPortfolioValues();
    }

    private void printPortfolioValues(){
        Map<AssetKey, Double> currentPricesForPortfolio = new HashMap<>();
        marketDataStreams.forEach((assetKey, stream) -> {
            RowVec kline = stream.peekNextKline();
            if (kline != null) currentPricesForPortfolio.put(assetKey, (Double)D.get(kline, 4));
        });
        agentWallets.forEach((agentId, wallet) -> {
            double portfolioValue = wallet.getTotalValue(referenceFiat, currentPricesForPortfolio);
            System.out.printf("Agent %s Portfolio Value: %.2f %s\n", agentId, portfolioValue, referenceFiat.baseAsset());
        });
    }

    private void processTick() {
        Map<AssetKey, RowVec> currentKlines = new HashMap<>();
        marketDataStreams.forEach((assetKey, stream) -> {
            if (stream.hasNext()) currentKlines.put(assetKey, stream.nextKline());
        });
        List<Future<?>> futures = new ArrayList<>();
        Map<String, SimWallet> updatedWallets = new HashMap<>(agentWallets);
        for (Join<TradingAgent, AssetKey> assignment : agentAssignments) {
            TradingAgent agent = assignment.f();
            AssetKey assetKey = assignment.s();
            SimWallet wallet = agentWallets.get(agent.getId());
            RowVec kline = currentKlines.get(assetKey);
            if (kline != null && wallet != null) {
                futures.add(executorService.submit(() -> {
                    AssetOutput output = agent.decide(assetKey, kline, wallet, dataHub.getSharedData());
                    // After decision, collect data for sharing if agent publishes
                    agent.publishData(output != null ? output.getReward() : 0.0);
                    synchronized (updatedWallets) {
                        processAgentAction(agent, assetKey, output, kline, updatedWallets.get(agent.getId()), updatedWallets);
                    }
                }));
            }
        }
        for (Future<?> future : futures) {
            try {
                future.get(); // Wait for all agent decisions to complete
            } catch (Exception e) {
                System.err.println("Error in agent decision: " + e.getMessage());
            }
        }
        agentWallets = updatedWallets; // Update wallets after all decisions are processed
    }

    private void processAgentAction(TradingAgent agent, AssetKey assetKey, AssetOutput action, RowVec kline, SimWallet wallet, Map<String, SimWallet> updatedWallets) {
        AssetMutation primaryAction = AssetMutation.HOLD_ACTION;
        if (action.get(AssetMutation.BUY_ACTION) > 0.5 && action.get(AssetMutation.BUY_ACTION) >= action.get(AssetMutation.SELL_ACTION)) {
            primaryAction = AssetMutation.BUY_ACTION;
        } else if (action.get(AssetMutation.SELL_ACTION) > 0.5 && action.get(AssetMutation.SELL_ACTION) > action.get(AssetMutation.BUY_ACTION)) {
            primaryAction = AssetMutation.SELL_ACTION;
        }
        if (primaryAction == AssetMutation.HOLD_ACTION) return;
        SimOrder.OrderSide side = (primaryAction == AssetMutation.BUY_ACTION) ? SimOrder.OrderSide.BUY : SimOrder.OrderSide.SELL;
        SimOrder.OrderType type = action.get(AssetMutation.AS_MARKET_ORDER) > 0.5 ? SimOrder.OrderType.MARKET : SimOrder.OrderType.LIMIT;
        double currentClose = (Double) D.get(kline, 4);
        double priceFraction = action.get(AssetMutation.PRICE_FRACTION);
        double orderPrice = (type == SimOrder.OrderType.LIMIT) ? currentClose * priceFraction : currentClose;
        double quantityFraction = action.get(AssetMutation.QUANTITY_FRACTION);
        double quantity;
        if (side == SimOrder.OrderSide.BUY) {
            double availableQuote = wallet.getQuoteBalance(assetKey);
            quantity = (availableQuote * quantityFraction) / orderPrice;
        } else {
            double availableBase = wallet.getBaseBalance(assetKey);
            quantity = availableBase * quantityFraction;
        }
        if (quantity < 0.00001) return;
        SimOrder order = new SimOrder(assetKey, side, type, quantity, orderPrice, currentTick, agent.getId());
        System.out.printf("[%s] %s: CREATED ORDER %s\n", agent.getId(), assetKey.toPairString(), order);
        boolean filled = false;
        double fillPrice = orderPrice;
        if (type == SimOrder.OrderType.MARKET) {
            filled = true;
        } else {
            double low = (Double) D.get(kline, 3);
            double high = (Double) D.get(kline, 2);
            if (side == SimOrder.OrderSide.BUY && orderPrice >= low) {
                fillPrice = Math.min(orderPrice, currentClose);
                filled = true;
            } else if (side == SimOrder.OrderSide.SELL && orderPrice <= high) {
                fillPrice = Math.max(orderPrice, currentClose);
                filled = true;
            }
        }
        if (filled) {
            Join<Boolean, SimWallet> result = wallet.applyTrade(side, assetKey, quantity, fillPrice);
            if (result.f()) {
                updatedWallets.put(agent.getId(), result.s());
                System.out.printf("[%s] %s: FILLED ORDER %s at price %.4f\n", agent.getId(), assetKey.toPairString(), order.type(), fillPrice);
            } else {
                System.out.printf("[%s] %s: FAILED TO FILL ORDER (Wallet rejected) %s at price %.4f\n", agent.getId(), assetKey.toPairString(), order.type(), fillPrice);
            }
        } else {
            System.out.printf("[%s] %s: UNFILLED LIMIT ORDER %s\n", agent.getId(), assetKey.toPairString(), order);
        }
    }
}
