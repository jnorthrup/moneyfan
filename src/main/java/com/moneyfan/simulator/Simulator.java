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
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class Simulator {
    private final Map<AssetKey, MarketDataStream> marketDataStreams = new TreeMap<>();
    private final List<Join<TradingAgent, AssetKey>> agentAssignments = Collections.synchronizedList(new ArrayList<>()); // Thread-safe list
    private Map<String, SimWallet> agentWallets = new ConcurrentHashMap<>(); // ConcurrentMap for thread-safe wallet updates
    private long currentTick = 0;
    private final AssetKey referenceFiat = AssetKey.of("USDT/USDT");
    private final ExecutorService executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
    private final AgentDataHub dataHub = new AgentDataHub();

    public void addMarketData(MarketDataStream stream) {
        marketDataStreams.put(stream.assetKey, stream);
    }

    public void registerAgent(TradingAgent agent, AssetKey assetToTrade, double initialBase, double initialQuote) {
        // Retrieve or create wallet, then initialize it immutably
        SimWallet currentWallet = agentWallets.computeIfAbsent(agent.getId(), SimWallet::new);
        SimWallet updatedWallet = currentWallet.initializeBalance(assetToTrade, initialBase, initialQuote);
        agentWallets.put(agent.getId(), updatedWallet); // Update the map with the new wallet instance

        agent.initialize(updatedWallet, assetToTrade);
        agentAssignments.add(D.jn(agent, assetToTrade));
        // No explicit registerAgent on dataHub; publishData is done by agent later
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
        executorService.shutdown(); // Clean up executor service
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

        // Get shared data snapshot BEFORE agents make decisions for this tick
        Series<Join<String, Double>> sharedDataSnapshot = dataHub.getSharedData();

        List<Callable<Void>> tasks = new ArrayList<>();
        // Use a temporary map for updates to prevent ConcurrentModificationException
        // and allow atomic swap at end of tick
        Map<String, SimWallet> newAgentWalletsState = new ConcurrentHashMap<>(agentWallets);

        for (Join<TradingAgent, AssetKey> assignment : agentAssignments) {
            final TradingAgent agent = assignment.f();
            final AssetKey assetKey = assignment.s();
            final SimWallet wallet = agentWallets.get(agent.getId()); // Get current wallet state
            final RowVec kline = currentKlines.get(assetKey);

            if (kline != null && wallet != null) {
                tasks.add(() -> {
                    // Agent decides based on current kline and shared data
                    AssetOutput output = agent.decide(assetKey, kline, wallet, sharedDataSnapshot);

                    // Agent publishes its reward/data
                    agent.publishData(output.getReward()); // FIXED: getReward() is now available

                    // Process agent action and update wallet state immutably
                    // Pass a reference to the newAgentWalletsState map for updates
                    processAgentAction(agent, assetKey, output, kline, wallet, newAgentWalletsState);
                    return null;
                });
            }
        }

        try {
            List<Future<Void>> futures = executorService.invokeAll(tasks);
            for (Future<Void> future : futures) {
                future.get(); // Wait for all tasks to complete and propagate exceptions
            }
        } catch (Exception e) {
            System.err.println("Error during parallel agent processing: " + e.getMessage());
            e.printStackTrace();
        }

        // Atomically update the main agentWallets map with the new state
        agentWallets = newAgentWalletsState;
        dataHub.reset(); // Clear shared data for the next tick
    }

    // This method now receives the mutable newAgentWalletsState map for updates
    private void processAgentAction(TradingAgent agent, AssetKey assetKey, AssetOutput action, RowVec kline, SimWallet currentAgentWallet, Map<String, SimWallet> newAgentWalletsState) {
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
            double availableQuote = currentAgentWallet.getQuoteBalance(assetKey);
            quantity = (availableQuote * quantityFraction) / orderPrice;
        } else {
            double availableBase = currentAgentWallet.getBaseBalance(assetKey);
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
            // Call applyTrade which now returns a Join<Boolean, SimWallet>
            Join<Boolean, SimWallet> tradeResult = currentAgentWallet.applyTrade(side, assetKey, quantity, fillPrice);
            if (tradeResult.f()) { // If trade was successful
                newAgentWalletsState.put(agent.getId(), tradeResult.s()); // Update with the new wallet instance
                System.out.printf("[%s] %s: FILLED ORDER %s at price %.4f\n", agent.getId(), assetKey.toPairString(), order.type(), fillPrice);
            } else {
                System.out.printf("[%s] %s: FAILED TO FILL ORDER (Wallet rejected) %s at price %.4f\n", agent.getId(), assetKey.toPairString(), order.type(), fillPrice);
            }
        } else {
            System.out.printf("[%s] %s: UNFILLED LIMIT ORDER %s\n", agent.getId(), assetKey.toPairString(), order);
        }
    }
}
