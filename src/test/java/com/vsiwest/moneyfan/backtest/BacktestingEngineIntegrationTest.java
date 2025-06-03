package com.vsiwest.moneyfan.backtest;

import com.vsiwest.moneyfan.ingestion.KlineData;
import com.vsiwest.moneyfan.strategy.MovingAverageCrossoverStrategy;
import com.vsiwest.moneyfan.strategy.TradingStrategy;
import com.vsiwest.moneyfan.strategy.Signal; // Though not directly used, good for context
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertFalse;

public class BacktestingEngineIntegrationTest {

    private static final double DELTA = 1e-5; // Tolerance for double comparisons

    // Helper to create KlineData easily, focusing only on time and close price.
    private KlineData createKline(long time, double closePrice) {
        return new KlineData(time, closePrice, closePrice, closePrice, closePrice, 0, time + 1000, 0, 0, 0, 0);
    }

    @Test
    void testRun_simpleScenario_MACO_oneBuyOneSell() {
        List<KlineData> klines = new ArrayList<>();
        // Data designed for MA(2,3) crossover
        // Short period: 2, Long period: 3
        // Strategy needs currentIndex >= 3 to generate signals (longPeriod)
        klines.add(createKline(1L, 10.0)); // idx 0
        klines.add(createKline(2L, 11.0)); // idx 1
        klines.add(createKline(3L, 10.0)); // idx 2. Prev S(2)=(11+10)/2=10.5. Prev L(3)=(10+11+10)/3=10.33
        klines.add(createKline(4L, 12.0)); // idx 3. Curr S(2)=(10+12)/2=11. Curr L(3)=(11+10+12)/3=11. BUY. (11>11 false, need adjustment)
                                        // Let's re-evaluate idx 3 for BUY:
                                        // Prev (idx 2): S2=(11+10)/2=10.5, L3=(10+11+10)/3=10.333
                                        // Curr (idx 3): S2=(10+12)/2=11,   L3=(11+10+12)/3=11
                                        // BUY if S_curr > L_curr AND S_prev <= L_prev
                                        // (11 > 11 is false). Let's make S_curr > L_curr
                                        // New Kline for idx 3: close 13
                                        // Curr (idx 3): S2=(10+13)/2=11.5, L3=(11+10+13)/3=11.333. BUY (11.5 > 11.333 && 10.5 > 10.333 (prev S > prev L - no, this is not buy))
                                        // Need S_prev <= L_prev for BUY.
                                        // Let klines[1] = 9.
                                        // idx 0: close 10
                                        // idx 1: close 9
                                        // idx 2: close 10. Prev S2=(9+10)/2=9.5. Prev L3=(10+9+10)/3=9.66. (S_prev <= L_prev TRUE)
                                        // idx 3: close 12. Curr S2=(10+12)/2=11. Curr L3=(9+10+12)/3=10.33. (S_curr > L_curr TRUE). BUY at 12.0
        klines.clear();
        klines.add(createKline(1L, 10.0)); // 0
        klines.add(createKline(2L, 9.0));  // 1
        klines.add(createKline(3L, 10.0)); // 2: Sprev=(9+10)/2=9.5. Lprev=(10+9+10)/3 = 9.66... (Sprev <= Lprev)
        klines.add(createKline(4L, 12.0)); // 3: Scurr=(10+12)/2=11. Lcurr=(9+10+12)/3 = 10.33... (Scurr > Lcurr). BUY triggered at 12.0

        klines.add(createKline(5L, 11.0)); // 4: Scurr=(12+11)/2=11.5. Lcurr=(10+12+11)/3=11. (Scurr > Lcurr). Hold.
                                           //    Sprev=11, Lprev=10.33. (Sprev > Lprev)
        klines.add(createKline(6L, 10.0)); // 5: Scurr=(11+10)/2=10.5. Lcurr=(12+11+10)/3=11. (Scurr < Lcurr). SELL triggered at 10.0
                                           //    Sprev=11.5, Lprev=11. (Sprev >= Lprev)
        klines.add(createKline(7L, 9.0));  // 6

        TradingStrategy strategy = new MovingAverageCrossoverStrategy(2, 3);
        BacktestingEngine engine = new BacktestingEngine();
        PerformanceReport report = engine.run(klines, strategy, 1000.0, 1.0);

        assertNotNull(report);
        assertEquals(1, report.getTrades().size(), "Should be one closed trade.");

        Trade trade = report.getTrades().get(0);
        assertEquals(Signal.BUY, trade.getSignal());
        assertEquals(12.0, trade.getEntryPrice(), DELTA); // Buy at close of kline index 3
        assertEquals(10.0, trade.getExitPrice(), DELTA);  // Sell at close of kline index 5
        assertEquals(-2.0, trade.getProfitOrLoss(), DELTA); // (10.0 - 12.0) * 1.0

        assertEquals(1, report.getTotalTrades());
        assertEquals(0, report.getWinningTrades());
        assertEquals(1, report.getLosingTrades());
        assertEquals(-2.0, report.getTotalProfitLoss(), DELTA);
        assertEquals(0.0, report.getProfitFactor(), DELTA); // 0 gross profit / 2 gross loss = 0
        assertEquals(0.0, report.getAverageWin(), DELTA);
        assertEquals(-2.0, report.getAverageLoss(), DELTA);
    }

    @Test
    void testRun_noTradesGenerated() {
        List<KlineData> klines = new ArrayList<>();
        // Prices consistently rise, short MA should stay above long MA after initial period
        klines.add(createKline(1L, 10.0));
        klines.add(createKline(2L, 11.0));
        klines.add(createKline(3L, 12.0));
        klines.add(createKline(4L, 13.0));
        klines.add(createKline(5L, 14.0));
        klines.add(createKline(6L, 15.0));

        TradingStrategy strategy = new MovingAverageCrossoverStrategy(2, 3);
        BacktestingEngine engine = new BacktestingEngine();
        PerformanceReport report = engine.run(klines, strategy, 1000.0, 1.0);

        assertNotNull(report);
        assertEquals(0, report.getTrades().size(), "Should be no trades.");
        assertEquals(0, report.getTotalTrades());
        assertEquals(0.0, report.getTotalProfitLoss(), DELTA);
    }

    @Test
    void testRun_positionOpenAtEnd_MACO() {
        List<KlineData> klines = new ArrayList<>();
        // Same data as the BUY signal scenario, but ending before a SELL signal
        klines.add(createKline(1L, 10.0)); // 0
        klines.add(createKline(2L, 9.0));  // 1
        klines.add(createKline(3L, 10.0)); // 2: Sprev=9.5. Lprev=9.66...
        klines.add(createKline(4L, 12.0)); // 3: Scurr=11. Lcurr=10.33... BUY triggered at 12.0 (close of this kline)
        klines.add(createKline(5L, 13.0)); // 4: This is the last kline. Position force-closed at 13.0.

        TradingStrategy strategy = new MovingAverageCrossoverStrategy(2, 3);
        BacktestingEngine engine = new BacktestingEngine();
        PerformanceReport report = engine.run(klines, strategy, 1000.0, 1.0);

        assertNotNull(report);
        assertEquals(1, report.getTrades().size(), "Should be one trade.");

        Trade trade = report.getTrades().get(0);
        assertEquals(Signal.BUY, trade.getSignal());
        assertEquals(12.0, trade.getEntryPrice(), DELTA); // Entry at close of kline index 3
        assertFalse(trade.isOpen(), "Trade should be closed at the end.");
        assertEquals(13.0, trade.getExitPrice(), DELTA);  // Exit at close of last kline (index 4)
        assertEquals(1005L, trade.getExitTime()); // exitTime = kline[4].time (5L) + 1000L = 1005L

        assertEquals(1.0, trade.getProfitOrLoss(), DELTA); // (13.0 - 12.0) * 1.0

        assertEquals(1, report.getTotalTrades());
        assertEquals(1, report.getWinningTrades());
        assertEquals(0, report.getLosingTrades());
        assertEquals(1.0, report.getTotalProfitLoss(), DELTA);
        assertEquals(Double.POSITIVE_INFINITY, report.getProfitFactor(), DELTA);
        assertEquals(1.0, report.getAverageWin(), DELTA);
        assertEquals(0.0, report.getAverageLoss(), DELTA);
    }
}
