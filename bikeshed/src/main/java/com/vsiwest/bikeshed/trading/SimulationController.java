package com.vsiwest.bikeshed.trading;

import com.example.bikeshed.dsel.Cursor;
import com.example.bikeshed.dsel.D;
import com.example.bikeshed.dsel.Series;

import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;
import java.util.concurrent.Future;

/**
 * Orchestrates the trading simulation, managing data flow between components.
 * This class demonstrates concurrent trade agents operating on a shared, immutable timeline.
 */
public class SimulationController {
    private final MarketDataSource dataSource;
    private final ObservationBuilder observationBuilder;
    private final AgentInterface[] agents; // Multiple agents
    private final ExecutionEngine executionEngine;
    private final MarketHistoryProvider historyProvider;
    private final PriceOracle priceOracle;

    private final ExecutorService agentDecisionExecutor;

    public SimulationController(MarketDataSource dataSource,
                                ObservationBuilder observationBuilder,
                                AgentInterface[] agents,
                                ExecutionEngine executionEngine,
                                MarketHistoryProvider historyProvider,
                                PriceOracle priceOracle) {
        this.dataSource = dataSource;
        this.observationBuilder = observationBuilder;
        this.agents = agents; // Array of agents for concurrent execution
        this.executionEngine = executionEngine;
        this.historyProvider = historyProvider;
        this.priceOracle = priceOracle;

        // Use a fixed thread pool for concurrent agent decision making.
        // Each agent can decide its action in parallel.
        this.agentDecisionExecutor = Executors.newFixedThreadPool(agents.length);
    }

    /**
     * Runs the trading simulation from start to finish.
     *
     * @param initialWalletState The initial wallet state for all agents.
     * @throws IOException If there's an issue with data source.
     */
    public void runSimulation(WalletState initialWalletState) throws IOException {
        System.out.println("Starting simulation...");
        // Initialize all agents' wallets to the same state for simplicity, or provide separate states.
        // For simplicity, ExecutionEngine manages a single wallet state here, which implies agents
        // are collaborating on a single fund. If agents are independent, each would have its own wallet.
        executionEngine.initialize(initialWalletState);

        MarketTick tick;
        int tickCount = 0;

        try (MarketDataSource ds = dataSource) { // Ensures dataSource is closed
            while ((tick = ds.nextTick()) != null) {
                tickCount++;
                System.out.println("Processing tick: " + tick.timestamp());
                historyProvider.addTick(tick); // Update shared historical timeline

                // Concurrent agent decision making
                // Using Future to collect results from parallel agent decisions
                Future<double[]>[] agentActions = new Future[agents.length];

                for (int i = 0; i < agents.length; i++) {
                    final int agentIdx = i;
                    // Agents access the shared timeline (historyProvider) for their observations.
                    // The observation itself is an immutable DSEL Cursor.
                    agentActions[i] = agentDecisionExecutor.submit(() -> {
                        Cursor observation = observationBuilder.buildObservation(tick, historyProvider);
                        // Agent decides action using the observation
                        return agents[agentIdx].decideAction(observation);
                    });
                }

                // Collect agent actions and process them sequentially (or combine for a single action)
                // For simplicity, let's assume we combine actions into one "master" action for execution.
                // Or, if agents are independent, they would each submit their own trades to the engine.
                // Here, we'll demonstrate a single aggregated action (e.g., average of all agent actions).
                double[] combinedAction = new double[1]; // Assuming a single action dimension (e.g., BTC_USD exposure)

                for (int i = 0; i < agents.length; i++) {
                    try {
                        double[] action = agentActions[i].get(); // Blocking call to get result
                        // Simple aggregation: sum actions. In reality, this would be a policy fusion.
                        if (action.length > 0) {
                            combinedAction[0] += action[0];
                        }
                    } catch (Exception e) {
                        System.err.println("Agent " + i + " failed to decide action: " + e.getMessage());
                        e.printStackTrace();
                    }
                }

                // Execute the combined action
                ExecutionResult result = executionEngine.processTick(tick, combinedAction);
                System.out.println("  Executed. PnL Change: " + String.format("%.2f", result.pnlChange()) + ", New Wallet: " + result.newWalletState().balances());

                // Agents learn from the execution result. This could also be concurrent.
                for (AgentInterface agent : agents) {
                    agent.learnFromExecution(result);
                }

                // Example of reward-based agent visibility (conceptual)
                // This would involve evaluating agent performance and dynamically adjusting
                // what data/markets they can "see" in subsequent observations.
                // For instance, if agent A has negative PnL for 10 consecutive ticks,
                // it might be restricted from trading certain assets.
                // This would mean the `ObservationBuilder` would need context about agent performance.
                // `D.filter` and `D.map` could be used to build predicates on performance records.
                // Example (pseudo-code):
                // Cursor agentPerformanceRecords = D.sr(agents.length, i -> agents[i].getPerformanceRecord());
                // Series<Boolean> profitableAgents = D.filter(agentPerformanceRecords,
                //     record -> ((Double)record.getValue(record.getColumnName("PnL")) > 0.0)
                // ).map(row -> true);
                // (ObservationBuilder could take `profitableAgents` to narrow data scope for some agents).
            }
        } catch (Exception e) {
            System.err.println("Simulation interrupted: " + e.getMessage());
            e.printStackTrace();
        } finally {
            agentDecisionExecutor.shutdown();
            try {
                if (!agentDecisionExecutor.awaitTermination(5, TimeUnit.SECONDS)) {
                    agentDecisionExecutor.shutdownNow();
                }
            } catch (InterruptedException e) {
                agentDecisionExecutor.shutdownNow();
                Thread.currentThread().interrupt();
            }
            System.out.println("Simulation finished. Total ticks: " + tickCount);
        }
    }
}
