package com.vsiwest.moneyfan.ingestion;

import com.vsiwest.moneyfan.backtest.BacktestingEngine;
import com.vsiwest.moneyfan.backtest.PerformanceReport;
import com.vsiwest.moneyfan.backtest.Trade; // For printing trades
import com.vsiwest.moneyfan.strategy.MovingAverageCrossoverStrategy;
import com.vsiwest.moneyfan.strategy.TradingStrategy;

import java.util.ArrayList;
import java.util.List;

public class TheEnvReadingSimulationHarnass {

    // Helper to create KlineData easily.
    private static KlineData createKline(long time, double closePrice) {
        // Using closePrice for open, high, low for simplicity in this example
        return new KlineData(time, closePrice, closePrice, closePrice, closePrice,
                             1000, time + 60000, 1000 * closePrice, 100,
                             500, 500 * closePrice);
    }

    public static void main(String[] args) {
        System.out.println("Starting TheEnvReadingSimulationHarnass...");

        // 1. Create sample KlineData
        List<KlineData> klines = new ArrayList<>();
        // цены закрытия: 100, 102, 104, 106, 108, 110, 108, 106, 104, 102, 100, 98, 96, 98, 100, 102, 104, 106, 108, 110
        klines.add(createKline(1L, 100));
        klines.add(createKline(2L, 102));
        klines.add(createKline(3L, 104));
        klines.add(createKline(4L, 106));
        klines.add(createKline(5L, 108)); // MA5 starts here (idx 4)
        klines.add(createKline(6L, 110));
        klines.add(createKline(7L, 108));
        klines.add(createKline(8L, 106));
        klines.add(createKline(9L, 104));
        klines.add(createKline(10L, 102)); // MA10 starts here (idx 9)
        klines.add(createKline(11L, 100));
        klines.add(createKline(12L, 98));
        klines.add(createKline(13L, 96));
        klines.add(createKline(14L, 98));
        klines.add(createKline(15L, 100));
        klines.add(createKline(16L, 102));
        klines.add(createKline(17L, 104));
        klines.add(createKline(18L, 106));
        klines.add(createKline(19L, 108));
        klines.add(createKline(20L, 110));


        // 2. Instantiate TradingStrategy
        // Strategy needs MA_long_period (10) data points to calculate previous MA.
        // So, signals will start from index 10.
        TradingStrategy strategy = new MovingAverageCrossoverStrategy(5, 10); // Short:5, Long:10

        // 3. Instantiate BacktestingEngine
        BacktestingEngine engine = new BacktestingEngine();

        // 4. Define parameters
        double initialBalance = 100000.0;
        double tradeSize = 10.0; // e.g., 10 units of the asset

        System.out.println("Running backtest with MovingAverageCrossoverStrategy(5, 10)...");
        // 5. Run backtest
        PerformanceReport report = engine.run(klines, strategy, initialBalance, tradeSize);

        // 6. Print report
        System.out.println("\n--- Performance Report ---");
        System.out.println(report);

        System.out.println("\n--- Trades Executed ---");
        if (report.getTrades().isEmpty()) {
            System.out.println("No trades were executed.");
        } else {
            for (Trade trade : report.getTrades()) {
                System.out.println(trade);
            }
        }

        System.out.println("\nTheEnvReadingSimulationHarnass executed successfully.");
    }
}
