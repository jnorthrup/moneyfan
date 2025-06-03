package com.vsiwest.moneyfan.strategy;

import com.vsiwest.moneyfan.ingestion.KlineData;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.ArrayList;
import java.util.List;

public class MovingAverageCrossoverStrategyTest {

    // Helper to create KlineData easily, focusing only on time and close price for these tests
    private KlineData createKline(long time, double closePrice) {
        // open, high, low, volume, quoteVolume, trades, takerBase, takerQuote are not used by this strategy's SMA.
        return new KlineData(time, closePrice, closePrice, closePrice, closePrice, 0, time + 1, 0, 0, 0, 0);
    }

    @Test
    void testConstructor_validPeriods() {
        MovingAverageCrossoverStrategy strategy = new MovingAverageCrossoverStrategy(5, 10);
        // No assertion needed, just checking no exception is thrown
    }

    @Test
    void testConstructor_invalidPeriods_shortNotLessThanLong() {
        assertThrows(IllegalArgumentException.class, () -> {
            new MovingAverageCrossoverStrategy(10, 5);
        });
        assertThrows(IllegalArgumentException.class, () -> {
            new MovingAverageCrossoverStrategy(10, 10);
        });
    }

    @Test
    void testConstructor_invalidPeriods_notPositive() {
        assertThrows(IllegalArgumentException.class, () -> {
            new MovingAverageCrossoverStrategy(0, 5);
        });
         assertThrows(IllegalArgumentException.class, () -> {
            new MovingAverageCrossoverStrategy(5, 0);
        });
    }


    @Test
    void testGenerateSignal_insufficientData() {
        MovingAverageCrossoverStrategy strategy = new MovingAverageCrossoverStrategy(3, 5);
        List<KlineData> klines = new ArrayList<>();
        for (int i = 0; i < 4; i++) { // Not enough for long period (5)
            klines.add(createKline(1000L + i, 10.0 + i));
        }
        // Current index 3, needs 5 data points for current long MA (indices 0,1,2,3,4)
        // and 5 for previous long MA (indices -1,0,1,2,3) - logic is currentIndex >= longPeriod
        assertEquals(Signal.HOLD, strategy.generateSignal(klines, 3)); // Not enough: 3 < 5 is true
        assertEquals(Signal.HOLD, strategy.generateSignal(klines, 4)); // Still not enough: 4 < 5 is true (needs index 0..4 for current, -1..3 for prev)
                                                                        // Actually, for index 4, current long MA uses 0,1,2,3,4. Prev long MA uses -1,0,1,2,3.
                                                                        // The condition is currentIndex < longPeriod, so at index 4 (0-based), it's 4 < 5, so HOLD.
    }

    @Test
    void testGenerateSignal_buySignal() {
        MovingAverageCrossoverStrategy strategy = new MovingAverageCrossoverStrategy(2, 3); // Short=2, Long=3
        List<KlineData> klines = new ArrayList<>();
        // Prices: 10, 10, 9, 12, 13 (indices 0-4)
        klines.add(createKline(1L, 10)); // idx 0
        klines.add(createKline(2L, 10)); // idx 1
        klines.add(createKline(3L, 9));  // idx 2. Prev Short MA(2)=(10+9)/2=9.5. Prev Long MA(3)=(10+10+9)/3=9.67
        klines.add(createKline(4L, 12)); // idx 3. Curr Short MA(2)=(9+12)/2=10.5. Curr Long MA(3)=(10+9+12)/3=10.33. BUY (10.5>10.33 curr, 9.5<=9.67 prev)
        klines.add(createKline(5L, 13)); // idx 4

        // Signal at index 3
        // Current MAs (index 3): prices 9, 12 for short; 10, 9, 12 for long
        //   Short MA = (9+12)/2 = 10.5
        //   Long MA = (10+9+12)/3 = 31/3 = 10.33
        // Previous MAs (index 2): prices 10, 9 for short; 10, 10, 9 for long
        //   Short MA = (10+9)/2 = 9.5
        //   Long MA = (10+10+9)/3 = 29/3 = 9.67
        // Check: short_curr (10.5) > long_curr (10.33)  -- TRUE
        // Check: short_prev (9.5) <= long_prev (9.67) -- TRUE
        // Expected: BUY
        assertEquals(Signal.BUY, strategy.generateSignal(klines, 3));
    }


    @Test
    void testGenerateSignal_sellSignal() {
        MovingAverageCrossoverStrategy strategy = new MovingAverageCrossoverStrategy(2, 3); // Short=2, Long=3
        List<KlineData> klines = new ArrayList<>();
        // Prices: 10, 11, 12, 9, 8 (indices 0-4)
        klines.add(createKline(1L, 10)); // idx 0
        klines.add(createKline(2L, 11)); // idx 1
        klines.add(createKline(3L, 12)); // idx 2. Prev Short MA(2)=(11+12)/2=11.5. Prev Long MA(3)=(10+11+12)/3=11
        klines.add(createKline(4L, 9));  // idx 3. Curr Short MA(2)=(12+9)/2=10.5. Curr Long MA(3)=(11+12+9)/3=10.67. SELL (10.5<10.67 curr, 11.5>=11 prev)
        klines.add(createKline(5L, 8));  // idx 4

        // Signal at index 3
        // Current MAs (index 3): prices 12, 9 for short; 11, 12, 9 for long
        //   Short MA = (12+9)/2 = 10.5
        //   Long MA = (11+12+9)/3 = 32/3 = 10.67
        // Previous MAs (index 2): prices 11, 12 for short; 10, 11, 12 for long
        //   Short MA = (11+12)/2 = 11.5
        //   Long MA = (10+11+12)/3 = 33/3 = 11
        // Check: short_curr (10.5) < long_curr (10.67) -- TRUE
        // Check: short_prev (11.5) >= long_prev (11)    -- TRUE
        // Expected: SELL
        assertEquals(Signal.SELL, strategy.generateSignal(klines, 3));
    }

    @Test
    void testGenerateSignal_holdSignal_noCrossover() {
        MovingAverageCrossoverStrategy strategy = new MovingAverageCrossoverStrategy(2, 3);
        List<KlineData> klines = new ArrayList<>();
        // Prices: 10, 11, 12, 13, 14 (short MA always above long MA, no recent crossover)
        klines.add(createKline(1L, 10)); // idx 0
        klines.add(createKline(2L, 11)); // idx 1
        klines.add(createKline(3L, 12)); // idx 2. Prev Short MA(2)=(11+12)/2=11.5. Prev Long MA(3)=(10+11+12)/3=11
        klines.add(createKline(4L, 13)); // idx 3. Curr Short MA(2)=(12+13)/2=12.5. Curr Long MA(3)=(11+12+13)/3=12
        klines.add(createKline(5L, 14)); // idx 4

        // Signal at index 3
        // Current MAs (index 3): Short (12.5), Long (12)
        // Previous MAs (index 2): Short (11.5), Long (11)
        // Check: short_curr (12.5) > long_curr (12) -- TRUE
        // Check: short_prev (11.5) <= long_prev (11) -- FALSE (11.5 > 11)
        // Expected: HOLD
        assertEquals(Signal.HOLD, strategy.generateSignal(klines, 3));
    }
     @Test
    void testGenerateSignal_holdSignal_MAsAreEqual() {
        MovingAverageCrossoverStrategy strategy = new MovingAverageCrossoverStrategy(2, 3);
        List<KlineData> klines = new ArrayList<>();
        // Prices carefully chosen so MAs become equal without prior crossover condition
        klines.add(createKline(1L, 10)); // idx 0
        klines.add(createKline(2L, 10)); // idx 1
        klines.add(createKline(3L, 10)); // idx 2. Prev S:(10+10)/2=10. Prev L:(10+10+10)/3=10
        klines.add(createKline(4L, 10)); // idx 3. Curr S:(10+10)/2=10. Curr L:(10+10+10)/3=10
        klines.add(createKline(5L, 10)); // idx 4

        // Signal at index 3
        // Current MAs (index 3): Short (10), Long (10)
        // Previous MAs (index 2): Short (10), Long (10)
        // Check: short_curr (10) > long_curr (10) -- FALSE
        // Check: short_curr (10) < long_curr (10) -- FALSE
        // Expected: HOLD
        assertEquals(Signal.HOLD, strategy.generateSignal(klines, 3));
    }
}
