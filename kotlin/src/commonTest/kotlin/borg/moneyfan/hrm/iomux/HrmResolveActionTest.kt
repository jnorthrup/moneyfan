package borg.moneyfan.hrm.iomux

import kotlin.test.Test
import kotlin.test.assertEquals

class HrmResolveActionTest {

    private val defaultMuxer = HrmTrikeShedIOMuxer()
    private val defaultConfig = HrmIOMuxerConfig()

    @Test
    fun score_within_hold_band_results_in_hold() {
        // defaultConfig.holdBand is 0.04
        val volatility = defaultConfig.minVolatilityForAction + 0.1

        assertEquals(HrmAction.HOLD, defaultMuxer.resolveAction(0.0, volatility))
        assertEquals(HrmAction.HOLD, defaultMuxer.resolveAction(0.039, volatility))
        assertEquals(HrmAction.HOLD, defaultMuxer.resolveAction(-0.039, volatility))
    }

    @Test
    fun low_volatility_results_in_hold() {
        // defaultConfig.minVolatilityForAction is 0.0015
        val highPositiveScore = 0.5
        val highNegativeScore = -0.5
        val lowVolatility = defaultConfig.minVolatilityForAction - 0.0001

        assertEquals(HrmAction.HOLD, defaultMuxer.resolveAction(highPositiveScore, lowVolatility))
        assertEquals(HrmAction.HOLD, defaultMuxer.resolveAction(highNegativeScore, lowVolatility))
    }

    @Test
    fun score_meeting_buy_threshold_results_in_buy() {
        // defaultConfig.buyThreshold is 0.14
        val volatility = defaultConfig.minVolatilityForAction + 0.1

        assertEquals(HrmAction.BUY, defaultMuxer.resolveAction(0.14, volatility))
        assertEquals(HrmAction.BUY, defaultMuxer.resolveAction(0.5, volatility))
    }

    @Test
    fun score_meeting_sell_threshold_results_in_sell() {
        // defaultConfig.sellThreshold is -0.14
        val volatility = defaultConfig.minVolatilityForAction + 0.1

        assertEquals(HrmAction.SELL, defaultMuxer.resolveAction(-0.14, volatility))
        assertEquals(HrmAction.SELL, defaultMuxer.resolveAction(-0.5, volatility))
    }

    @Test
    fun score_between_hold_band_and_thresholds_results_in_hold() {
        // holdBand=0.04, buyThreshold=0.14, sellThreshold=-0.14
        val volatility = defaultConfig.minVolatilityForAction + 0.1

        assertEquals(HrmAction.HOLD, defaultMuxer.resolveAction(0.10, volatility))
        assertEquals(HrmAction.HOLD, defaultMuxer.resolveAction(-0.10, volatility))
    }

    @Test
    fun custom_config_affects_resolution() {
        val customConfig = HrmIOMuxerConfig(
            buyThreshold = 0.30,
            sellThreshold = -0.30,
            holdBand = 0.10,
            minVolatilityForAction = 0.05
        )
        val muxer = HrmTrikeShedIOMuxer()
        muxer.setConfig(customConfig)

        // Hold band
        assertEquals(HrmAction.HOLD, muxer.resolveAction(0.08, 0.1))

        // Low volatility
        assertEquals(HrmAction.HOLD, muxer.resolveAction(0.5, 0.04))

        // Thresholds
        assertEquals(HrmAction.BUY, muxer.resolveAction(0.30, 0.1))
        assertEquals(HrmAction.SELL, muxer.resolveAction(-0.30, 0.1))

        // Between hold and threshold
        assertEquals(HrmAction.HOLD, muxer.resolveAction(0.25, 0.1))
    }
}
