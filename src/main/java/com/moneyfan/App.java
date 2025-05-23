package com.moneyfan;

import com.moneyfan.dsel.D;
import com.moneyfan.dsel.core.Cursor;
import com.moneyfan.simulator.MarketDataStream;
import com.moneyfan.simulator.SimWallet;
import com.moneyfan.simulator.Simulator;
import com.moneyfan.simulator.agent.SimpleRuleAgent;
import com.moneyfan.simulator.agent.TradingAgent;
import com.moneyfan.simulator.model.AssetKey;
import com.moneyfan.util.DataUtil;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class App {
    public static void main(String[] args) {
        System.out.println("Trading Simulator with DSEL");

        String baseImportDir = "mpdata_import"; // As per README, relative to project root
        String baseIsamDir = "mpdata_isam";   // Store ISAM files here
        long recordsPerPair = 200; // Number of kline records per dummy CSV
        long simulationTicks = 150; // How many ticks to run the simulation

        try {
            // Ensure directories exist (DataUtil.ensureDirs will be called by prepareDataForPairs)
            Files.createDirectories(Paths.get(baseImportDir));
            Files.createDirectories(Paths.get(baseIsamDir));

            // 1. Prepare Data (Generate dummy CSVs and convert to ISAM)
            DataUtil.prepareDataForPairs(baseImportDir, baseIsamDir, DataUtil.NINETEEN_PAIRS, recordsPerPair);

            // 2. Initialize Simulator
            Simulator simulator = new Simulator();

            // 3. Load ISAM data and register agents
            for (AssetKey assetKey : DataUtil.NINETEEN_PAIRS) {
                Path isamDir = DataUtil.ensureDirs(baseIsamDir, assetKey); // Get the correct ISAM dir
                String isamPathBase = isamDir.resolve("final-" + assetKey.baseAsset() + "-" + assetKey.quoteAsset() + "-1m").toString();
                try {
                    D.IsamCursor klineIsamCursor = new D.IsamCursor(isamPathBase);
                    MarketDataStream stream = new MarketDataStream(assetKey, klineIsamCursor);
                    simulator.addMarketData(stream);
                    System.out.println("Loaded ISAM for " + assetKey.toPairString() + " with " + D.sz(klineIsamCursor) + " records.");

                    // Register an agent for this asset
                    TradingAgent agent = new SimpleRuleAgent("Agent_" + assetKey.toSymbol());
                    simulator.registerAgent(agent, assetKey, 0, 10000); // Initial: 0 base, 10000 quote
                } catch (IOException e) {
                    System.err.println("Failed to load ISAM data for " + assetKey.toPairString() + ": " + e.getMessage());
                }
            }

            // 4. Run Simulation
            if (!DataUtil.NINETEEN_PAIRS.isEmpty()) { // Only run if data was loaded
                 simulator.runSimulation(simulationTicks);
            } else {
                System.out.println("No data loaded, simulation cannot run.");
            }

        } catch (IOException e) {
            System.err.println("Error during simulation setup or run: " + e.getMessage());
            e.printStackTrace();
        } finally {
            // Optional: Clean up dummy data directories after run
            // try { deleteDirectoryRecursively(Paths.get(baseImportDir)); } catch (Exception e) {}
            // try { deleteDirectoryRecursively(Paths.get(baseIsamDir)); } catch (Exception e) {}
        }
    }
    // Helper to clean up, if needed
    // static void deleteDirectoryRecursively(Path path) throws IOException { ... }
}
