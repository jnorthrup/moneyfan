package com.vsiwest.moneyfan.backtest;

import com.vsiwest.moneyfan.ingestion.KlineData;
import com.vsiwest.moneyfan.strategy.MovingAverageCrossoverStrategy; // For main method test
import com.vsiwest.moneyfan.strategy.TradingStrategy;
import com.vsiwest.moneyfan.strategy.Signal;

import java.util.List;
import java.util.ArrayList;

public class BacktestingEngine {

    /**
     * Runs a backtest for a given strategy on a list of Kline data.
     *
     * @param klines         The historical Kline data.
     * @param strategy       The trading strategy to test.
     * @param initialBalance The starting balance for the backtest.
     * @param tradeSize      The fixed size for each trade (e.g., number of units of the asset).
     * @return A PerformanceReport summarizing the results of the backtest.
     */
    public PerformanceReport run(List<KlineData> klines, TradingStrategy strategy, double initialBalance, double tradeSize) {
        if (klines == null || klines.isEmpty()) {
            throw new IllegalArgumentException("Kline data cannot be null or empty.");
        }
        if (strategy == null) {
            throw new IllegalArgumentException("Trading strategy cannot be null.");
        }
        if (tradeSize <= 0) {
            throw new IllegalArgumentException("Trade size must be positive.");
        }

        List<Trade> executedTrades = new ArrayList<>();
        // double currentBalance = initialBalance; // Balance tracking can be added later if P&L is tied to it.
                                                // For now, P&L is calculated per trade independently of balance.
        Trade openPosition = null;

        strategy.init(klines); // Allow strategy to initialize

        for (int i = 0; i < klines.size(); i++) {
            KlineData currentKline = klines.get(i);
            // The effective price for a signal at currentIndex is typically the open of the next bar,
            // or close of the current bar if acting on market-on-close.
            // For simplicity, using current kline's close price.
            double currentPrice = currentKline.getClosePrice();
            long currentTime = currentKline.getCloseTime(); // Or openTime, depending on convention

            Signal signal = strategy.generateSignal(klines, i);

            if (openPosition == null) { // No open position
                if (signal == Signal.BUY) {
                    openPosition = new Trade(Signal.BUY, currentPrice, currentTime, tradeSize);
                    executedTrades.add(openPosition);
                    // System.out.println("[" + currentTime + "] Opened BUY at " + currentPrice);
                }
                // If signal is SELL and no open position, we could open a short position here.
                // For now, we only handle long positions.
            } else { // Have an open position
                if (openPosition.getSignal() == Signal.BUY) { // Currently long
                    // Exit long position if SELL signal, or if it's the last kline (force close)
                    if (signal == Signal.SELL) {
                        openPosition.close(currentPrice, currentTime);
                        // currentBalance += openPosition.getProfitOrLoss(); // If tracking balance
                        // System.out.println("[" + currentTime + "] Closed BUY at " + currentPrice + " P&L: " + openPosition.getProfitOrLoss());
                        openPosition = null;
                    }
                }
                // Add logic for managing open SHORT positions if shorting is implemented
            }
        }

        // If a position is still open at the end of the data series, close it
        if (openPosition != null && openPosition.isOpen()) {
            KlineData lastKline = klines.get(klines.size() - 1);
            openPosition.close(lastKline.getClosePrice(), lastKline.getCloseTime());
            // currentBalance += openPosition.getProfitOrLoss(); // If tracking balance
            // System.out.println("[End] Force closed position at " + lastKline.getClosePrice() + " P&L: " + openPosition.getProfitOrLoss());
        }

        return new PerformanceReport(executedTrades);
    }

    public static void main(String[] args) {
        // Create sample KlineData
        List<KlineData> sampleKlines = new ArrayList<>();
        // Format: openTime, open, high, low, close, volume, closeTime, quoteVolume, trades, takerBase, takerQuote
        sampleKlines.add(new KlineData(1000L, 10, 12, 9, 11, 100, 1001L, 1100, 10, 50, 550));   // idx 0
        sampleKlines.add(new KlineData(1001L, 11, 13, 10, 12, 110, 1002L, 1320, 12, 60, 720));  // idx 1
        sampleKlines.add(new KlineData(1002L, 12, 14, 11, 13, 120, 1003L, 1560, 13, 70, 910));  // idx 2
        sampleKlines.add(new KlineData(1003L, 13, 15, 12, 14, 130, 1004L, 1820, 14, 80, 1120)); // idx 3
        sampleKlines.add(new KlineData(1004L, 14, 16, 13, 15, 140, 1005L, 2100, 15, 90, 1350)); // idx 4
        sampleKlines.add(new KlineData(1005L, 15, 17, 14, 12, 150, 1006L, 1800, 16, 70, 840));  // idx 5 Close drops
        sampleKlines.add(new KlineData(1006L, 12, 13, 10, 11, 160, 1007L, 1760, 17, 60, 660));  // idx 6
        sampleKlines.add(new KlineData(1007L, 11, 12, 9, 10, 170, 1008L, 1700, 18, 50, 500));   // idx 7
        sampleKlines.add(new KlineData(1008L, 10, 11, 8, 13, 180, 1009L, 2340, 19, 90, 1170));  // idx 8 Close rallies
        sampleKlines.add(new KlineData(1009L, 13, 15, 12, 14, 190, 1010L, 2660, 20, 100, 1400)); // idx 9

        // Instantiate a strategy
        TradingStrategy maStrategy = new MovingAverageCrossoverStrategy(3, 5); // Short:3, Long:5

        BacktestingEngine engine = new BacktestingEngine();
        double initialBalance = 10000.0;
        double tradeAmount = 1.0; // Trade 1 unit of asset

        System.out.println("Starting backtest with MovingAverageCrossoverStrategy(3, 5)...");
        PerformanceReport report = engine.run(sampleKlines, maStrategy, initialBalance, tradeAmount);

        System.out.println("\nBacktest Finished. Performance Report:");
        System.out.println(report);

        System.out.println("\nIndividual Trades:");
        if (report.getTrades().isEmpty()) {
            System.out.println("No trades were executed.");
        } else {
            for (Trade trade : report.getTrades()) {
                System.out.println(trade);
            }
        }
    }
}
