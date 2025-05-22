package org.bereft.adapter

import com.binance.api.client.domain.market.Candlestick
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import java.time.Instant

/**
 * Unit tests proving behaviour of [BinanceMarketDataAdapter].
 */
class BinanceAdaptersTest {

    /**
     * Given a populated [Candlestick] the adapter should expose the same numeric values converted
     * from strings to doubles and fall back to 0/NaN defaults when data is missing.
     */
    @Test
    fun `converts candlestick strings into MarketData doubles`() {
        val now = Instant.now().toEpochMilli()
        val candle = Candlestick().apply {
            openTime = now
            open = "100.0"
            high = "110.5"
            low = "95.3"
            close = "105.1"
            volume = "123.456"
            closeTime = now + 60_000
        }

        val md = candle.asMarketData()

        assertEquals(now, md.openTime)
        assertEquals(100.0, md.open)
        assertEquals(110.5, md.high)
        assertEquals(95.3, md.low)
        assertEquals(105.1, md.close)
        assertEquals(123.456, md.volume)
        assertEquals(now + 60_000, md.closeTime)
    }

    /**
     * Missing or malformed numeric strings should yield [Double.NaN] values so that callers can
     * easily filter out incomplete data rows.
     */
    @Test
    fun `malformed numbers return NaN`() {
        val candle = Candlestick().apply {
            open = "not-a-number"
        }

        val md = candle.asMarketData()
        assertTrue(md.open.isNaN())
        assertTrue(md.high.isNaN())
        assertTrue(md.low.isNaN())
        assertTrue(md.close.isNaN())
        assertTrue(md.volume.isNaN())
    }
}