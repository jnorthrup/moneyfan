package com.vsiwest.moneyfan.backtest;

import com.vsiwest.moneyfan.strategy.Signal;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class PerformanceReportTest {

    private static final double DELTA = 1e-5; // Tolerance for double comparisons

    // Helper to create a closed Trade object
    private Trade createClosedTrade(Signal signal, double entryPrice, double exitPrice, double size, long entryTime, long exitTime) {
        Trade trade = new Trade(signal, entryPrice, entryTime, size);
        trade.close(exitPrice, exitTime);
        return trade;
    }

    private Trade createOpenTrade(Signal signal, double entryPrice, double size, long entryTime) {
        return new Trade(signal, entryPrice, entryTime, size);
    }

    @Test
    void testCalculateMetrics_noTrades() {
        PerformanceReport report = new PerformanceReport(Collections.emptyList());

        assertEquals(0, report.getTotalTrades());
        assertEquals(0, report.getWinningTrades());
        assertEquals(0, report.getLosingTrades());
        assertEquals(0.0, report.getTotalProfitLoss(), DELTA);
        assertEquals(0.0, report.getProfitFactor(), DELTA); // Current impl: 0 if no trades or no wins/losses
        assertEquals(0.0, report.getAverageWin(), DELTA);
        assertEquals(0.0, report.getAverageLoss(), DELTA);
        assertNotNull(report.getTrades());
        assertEquals(0, report.getTrades().size());
    }

    @Test
    void testCalculateMetrics_onlyWinningTrades() {
        List<Trade> trades = new ArrayList<>();
        trades.add(createClosedTrade(Signal.BUY, 100, 110, 1, 1L, 2L)); // P&L = +10
        trades.add(createClosedTrade(Signal.BUY, 100, 120, 1, 3L, 4L)); // P&L = +20

        PerformanceReport report = new PerformanceReport(trades);

        assertEquals(2, report.getTotalTrades());
        assertEquals(2, report.getWinningTrades());
        assertEquals(0, report.getLosingTrades());
        assertEquals(30.0, report.getTotalProfitLoss(), DELTA);
        assertEquals(15.0, report.getAverageWin(), DELTA);
        assertEquals(0.0, report.getAverageLoss(), DELTA);
        // Profit factor: grossProfit / grossLoss. If grossLoss is 0, POSITIVE_INFINITY
        assertEquals(Double.POSITIVE_INFINITY, report.getProfitFactor(), DELTA);
    }

    @Test
    void testCalculateMetrics_onlyLosingTrades() {
        List<Trade> trades = new ArrayList<>();
        trades.add(createClosedTrade(Signal.BUY, 100, 90, 1, 1L, 2L));  // P&L = -10
        trades.add(createClosedTrade(Signal.SELL, 100, 110, 1, 3L, 4L)); // P&L = -10 (entry 100, exit 110 for SELL)

        PerformanceReport report = new PerformanceReport(trades);

        assertEquals(2, report.getTotalTrades());
        assertEquals(0, report.getWinningTrades());
        assertEquals(2, report.getLosingTrades());
        assertEquals(-20.0, report.getTotalProfitLoss(), DELTA);
        assertEquals(0.0, report.getAverageWin(), DELTA);
        assertEquals(-10.0, report.getAverageLoss(), DELTA);
        assertEquals(0.0, report.getProfitFactor(), DELTA); // GrossProfit is 0
    }

    @Test
    void testCalculateMetrics_mixedTrades() {
        List<Trade> trades = new ArrayList<>();
        trades.add(createClosedTrade(Signal.BUY, 100, 120, 1, 1L, 2L));  // P&L = +20 (Win)
        trades.add(createClosedTrade(Signal.BUY, 100, 90, 1, 3L, 4L));   // P&L = -10 (Loss)
        trades.add(createClosedTrade(Signal.SELL, 100, 95, 1, 5L, 6L));  // P&L = +5  (Win)
        trades.add(createClosedTrade(Signal.SELL, 100, 105, 1, 7L, 8L)); // P&L = -5  (Loss)
        trades.add(createClosedTrade(Signal.BUY, 100, 100, 1, 9L, 10L)); // P&L = 0 (Neutral - not counted as win or loss in current impl for avg calcs)


        PerformanceReport report = new PerformanceReport(trades);
        // Total 5 trades, but one is neutral. Neutral trades affect total P&L but not win/loss counts for averages.
        assertEquals(5, report.getTotalTrades());
        assertEquals(2, report.getWinningTrades()); // +20, +5
        assertEquals(2, report.getLosingTrades()); // -10, -5
        assertEquals(10.0, report.getTotalProfitLoss(), DELTA); // 20 - 10 + 5 - 5 + 0 = 10

        double expectedAverageWin = (20.0 + 5.0) / 2.0; // 12.5
        double expectedAverageLoss = (-10.0 - 5.0) / 2.0; // -7.5
        double expectedProfitFactor = (20.0 + 5.0) / (10.0 + 5.0); // 25 / 15 = 1.66666...

        assertEquals(expectedAverageWin, report.getAverageWin(), DELTA);
        assertEquals(expectedAverageLoss, report.getAverageLoss(), DELTA);
        assertEquals(expectedProfitFactor, report.getProfitFactor(), DELTA);
    }

    @Test
    void testCalculateMetrics_profitFactorSpecialCases() {
        // Case 1: Only winning trades (GrossLoss = 0)
        List<Trade> winningOnly = new ArrayList<>();
        winningOnly.add(createClosedTrade(Signal.BUY, 100, 110, 1, 1L, 2L));
        PerformanceReport reportWinning = new PerformanceReport(winningOnly);
        assertEquals(Double.POSITIVE_INFINITY, reportWinning.getProfitFactor(), DELTA);

        // Case 2: Only losing trades (GrossProfit = 0)
        List<Trade> losingOnly = new ArrayList<>();
        losingOnly.add(createClosedTrade(Signal.BUY, 100, 90, 1, 1L, 2L));
        PerformanceReport reportLosing = new PerformanceReport(losingOnly);
        assertEquals(0.0, reportLosing.getProfitFactor(), DELTA);

        // Case 3: No P&L trades (e.g., all trades break even)
        List<Trade> neutralOnly = new ArrayList<>();
        neutralOnly.add(createClosedTrade(Signal.BUY, 100, 100, 1, 1L, 2L));
        PerformanceReport reportNeutral = new PerformanceReport(neutralOnly);
        assertEquals(0.0, reportNeutral.getProfitFactor(), DELTA); // Both gross profit and gross loss are 0
    }

    @Test
    void testCalculateMetrics_openTradesIgnoredForPnlMetrics() {
        List<Trade> trades = new ArrayList<>();
        trades.add(createClosedTrade(Signal.BUY, 100, 110, 1, 1L, 2L)); // P&L = +10
        trades.add(createOpenTrade(Signal.BUY, 105, 1, 3L)); // Open trade

        PerformanceReport report = new PerformanceReport(trades);

        // PerformanceReport's calculateMetrics adjusts totalTrades if a trade is open and P&L is null.
        assertEquals(1, report.getTotalTrades()); // Only closed trade considered for P&L metrics count
        assertEquals(1, report.getWinningTrades());
        assertEquals(0, report.getLosingTrades());
        assertEquals(10.0, report.getTotalProfitLoss(), DELTA);
        assertEquals(10.0, report.getAverageWin(), DELTA);
        assertEquals(Double.POSITIVE_INFINITY, report.getProfitFactor(), DELTA);

        // To check original list size if needed:
        assertEquals(2, trades.size()); // Original list still has 2 trades
        assertEquals(2, report.getTrades().size()); // report.getTrades() returns a copy of the original list
    }
}
