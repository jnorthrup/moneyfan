package borg.moneyfan.hrm.iomux

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class HrmTrikeShedIOMuxerTest {

    @Test
    fun bullish_drift_trends_to_buy_signal() {
        val muxer = HrmTrikeShedIOMuxer()
        var decision: HrmMuxDecision? = null
        var price = 100.0

        repeat(120) { i ->
            val open = price
            val close = open + 0.45 + (i * 0.001)
            val high = close + 0.15
            val low = open - 0.12
            val volume = 1_000.0 + i
            decision = muxer.ingest(
                HrmIoFrame(
                    symbol = "BTC/USD",
                    open = open,
                    high = high,
                    low = low,
                    close = close,
                    volume = volume,
                    epochMillis = i.toLong(),
                )
            )
            price = close
        }

        val resolved = assertNotNull(decision)
        assertTrue(resolved.score > 0.0, "Expected positive aggregate score for bullish drift")
        assertEquals(HrmAction.BUY, resolved.action)
    }

    @Test
    fun bearish_drift_trends_to_sell_signal() {
        val muxer = HrmTrikeShedIOMuxer()
        var decision: HrmMuxDecision? = null
        var price = 100.0

        repeat(120) { i ->
            val open = price
            val close = open - 0.42 - (i * 0.001)
            val high = open + 0.10
            val low = close - 0.10
            val volume = 1_500.0 + (i * 2.0)
            decision = muxer.ingest(
                HrmIoFrame(
                    symbol = "ETH/USD",
                    open = open,
                    high = high,
                    low = low,
                    close = close,
                    volume = volume,
                    epochMillis = i.toLong(),
                )
            )
            price = close
        }

        val resolved = assertNotNull(decision)
        assertTrue(resolved.score < 0.0, "Expected negative aggregate score for bearish drift")
        assertEquals(HrmAction.SELL, resolved.action)
    }

    @Test
    fun symbol_states_are_isolated() {
        val muxer = HrmTrikeShedIOMuxer()
        val symbolA = "BTC/USD"
        val symbolB = "SOL/USD"

        repeat(40) { i ->
            val step = i.toDouble() * 0.2
            muxer.ingest(
                HrmIoFrame(
                    symbol = symbolA,
                    open = 100.0 + step,
                    high = 100.5 + step,
                    low = 99.8 + step,
                    close = 100.4 + step,
                    volume = 1000.0 + i,
                    epochMillis = i.toLong(),
                )
            )
        }

        val firstB = muxer.ingest(
            HrmIoFrame(
                symbol = symbolB,
                open = 60.0,
                high = 60.2,
                low = 59.8,
                close = 60.1,
                volume = 500.0,
                epochMillis = 1L,
            )
        )
        assertTrue(firstB.confidence >= 0.0)
        assertEquals(muxer.laneCount, firstB.laneSignals.size)
    }
}
