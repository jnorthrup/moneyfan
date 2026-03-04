package borg.moneyfan.hrm.codec

import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class SlotLineup {
    private val volatilityBreakout = CommonMainCodecModel(slotId = 1, slotName = "volatility_breakout")
    private val momentumTrend = CommonMainCodecModel(slotId = 2, slotName = "momentum_trend")
    private val meanReversion = CommonMainCodecModel(slotId = 3, slotName = "mean_reversion")
    private val trendFollowing = CommonMainCodecModel(slotId = 4, slotName = "trend_following")
    private val pairsTrading = CommonMainCodecModel(slotId = 5, slotName = "pairs_trading")
    private val gridTrading = CommonMainCodecModel(slotId = 6, slotName = "grid_trading")
    private val volumeProfile = CommonMainCodecModel(slotId = 7, slotName = "volume_profile")
    private val orderFlow = CommonMainCodecModel(slotId = 8, slotName = "order_flow")
    private val correlationTrading = CommonMainCodecModel(slotId = 9, slotName = "correlation_trading")
    private val liquidityMaking = CommonMainCodecModel(slotId = 10, slotName = "liquidity_making")
    private val sectorRotation = CommonMainCodecModel(slotId = 11, slotName = "sector_rotation")
    private val compositeAlpha = CommonMainCodecModel(slotId = 12, slotName = "composite_alpha")
    private val rsiReversal = CommonMainCodecModel(slotId = 13, slotName = "rsi_reversal")
    private val bollingerBands = CommonMainCodecModel(slotId = 14, slotName = "bollinger_bands")
    private val macdCrossover = CommonMainCodecModel(slotId = 15, slotName = "macd_crossover")
    private val stochasticKd = CommonMainCodecModel(slotId = 16, slotName = "stochastic_kd")
    private val adxTrendStrength = CommonMainCodecModel(slotId = 17, slotName = "adx_trend_strength")
    private val vwapMeanReversion = CommonMainCodecModel(slotId = 18, slotName = "vwap_mean_reversion")
    private val kalmanFilterTrend = CommonMainCodecModel(slotId = 19, slotName = "kalman_filter_trend")
    private val hurstRegime = CommonMainCodecModel(slotId = 20, slotName = "hurst_regime")
    private val randomForestClassifier = CommonMainCodecModel(slotId = 21, slotName = "random_forest_classifier")
    private val xgboostSignal = CommonMainCodecModel(slotId = 22, slotName = "xgboost_signal")
    private val transformerAttention = CommonMainCodecModel(slotId = 23, slotName = "transformer_attention")
    private val zscoreStatArb = CommonMainCodecModel(slotId = 24, slotName = "zscore_stat_arb")

    fun evaluateSlot(slotId: Int, input: CodecInput): CodecSignal {
        return slot(slotId).evaluate(input)
    }

    fun evaluateAll(input: CodecInput): List<CodecSignal> = listOf(
        volatilityBreakout.evaluate(input),
        momentumTrend.evaluate(input),
        meanReversion.evaluate(input),
        trendFollowing.evaluate(input),
        pairsTrading.evaluate(input),
        gridTrading.evaluate(input),
        volumeProfile.evaluate(input),
        orderFlow.evaluate(input),
        correlationTrading.evaluate(input),
        liquidityMaking.evaluate(input),
        sectorRotation.evaluate(input),
        compositeAlpha.evaluate(input),
        rsiReversal.evaluate(input),
        bollingerBands.evaluate(input),
        macdCrossover.evaluate(input),
        stochasticKd.evaluate(input),
        adxTrendStrength.evaluate(input),
        vwapMeanReversion.evaluate(input),
        kalmanFilterTrend.evaluate(input),
        hurstRegime.evaluate(input),
        randomForestClassifier.evaluate(input),
        xgboostSignal.evaluate(input),
        transformerAttention.evaluate(input),
        zscoreStatArb.evaluate(input),
    )

    fun reset() {
        volatilityBreakout.reset()
        momentumTrend.reset()
        meanReversion.reset()
        trendFollowing.reset()
        pairsTrading.reset()
        gridTrading.reset()
        volumeProfile.reset()
        orderFlow.reset()
        correlationTrading.reset()
        liquidityMaking.reset()
        sectorRotation.reset()
        compositeAlpha.reset()
        rsiReversal.reset()
        bollingerBands.reset()
        macdCrossover.reset()
        stochasticKd.reset()
        adxTrendStrength.reset()
        vwapMeanReversion.reset()
        kalmanFilterTrend.reset()
        hurstRegime.reset()
        randomForestClassifier.reset()
        xgboostSignal.reset()
        transformerAttention.reset()
        zscoreStatArb.reset()
    }

    fun slot(slotId: Int): CodecModel {
        return when (slotId) {
            1 -> volatilityBreakout
            2 -> momentumTrend
            3 -> meanReversion
            4 -> trendFollowing
            5 -> pairsTrading
            6 -> gridTrading
            7 -> volumeProfile
            8 -> orderFlow
            9 -> correlationTrading
            10 -> liquidityMaking
            11 -> sectorRotation
            12 -> compositeAlpha
            13 -> rsiReversal
            14 -> bollingerBands
            15 -> macdCrossover
            16 -> stochasticKd
            17 -> adxTrendStrength
            18 -> vwapMeanReversion
            19 -> kalmanFilterTrend
            20 -> hurstRegime
            21 -> randomForestClassifier
            22 -> xgboostSignal
            23 -> transformerAttention
            24 -> zscoreStatArb
            else -> throw IllegalArgumentException("slotId must be in 1..24")
        }
    }

    fun allModels(): List<CodecModel> = listOf(
        volatilityBreakout,
        momentumTrend,
        meanReversion,
        trendFollowing,
        pairsTrading,
        gridTrading,
        volumeProfile,
        orderFlow,
        correlationTrading,
        liquidityMaking,
        sectorRotation,
        compositeAlpha,
        rsiReversal,
        bollingerBands,
        macdCrossover,
        stochasticKd,
        adxTrendStrength,
        vwapMeanReversion,
        kalmanFilterTrend,
        hurstRegime,
        randomForestClassifier,
        xgboostSignal,
        transformerAttention,
        zscoreStatArb,
    )
}

private data class GridTradingParams(
    val window: Int = 20,
    val gridLevels: Int = 5,
    val atrBandMultiplier: Double = 2.0,
    val sigmaBandMultiplier: Double = 1.5,
    val minBandFloorPct: Double = 0.005,
    val baseConfidence: Double = 0.3,
    val gridSignalWeight: Double = 0.5,
    val edgeWeight: Double = 0.2,
    val oscillationBase: Double = 0.5,
    val oscillationSlope: Double = 1.0,
)

private val GRID_TRADING_PARAMS = GridTradingParams()

/**
 * Numeric reason codes emitted in `instruments["gate_reason_code"]`.
 * Negative values mean an early gate/reject path, positive values mean active path.
 */
private object GateReason {
    const val ACTIVE = 1.0
    const val UNKNOWN_CODEC = -999.0
    const val INSUFFICIENT_FEATURE_HISTORY = -1.0
    const val INSUFFICIENT_OHLCV_HISTORY = -2.0
    const val OUTSIDE_DYNAMIC_BAND = -3.0
    const val DEGENERATE_RANGE = -4.0
    const val NO_DIRECTIONAL_VOTES = -5.0
    const val NO_ALPHA_QUALITY = -6.0
}

private data class DecisionTree(
    val featureIdx: Int,
    val thresholds: DoubleArray,
    val directions: DoubleArray,
) {
    fun predict(features: DoubleArray): Double {
        val value = if (featureIdx < features.size) features[featureIdx] else 0.0
        var i = 0
        while (i < thresholds.size) {
            if (value <= thresholds[i]) return directions[i]
            i += 1
        }
        return directions[directions.lastIndex]
    }
}

private data class WeakLearner(
    val featureIdx: Int,
    val threshold: Double,
    val left: Double,
    val right: Double,
) {
    fun predict(features: DoubleArray): Double {
        val value = if (featureIdx < features.size) features[featureIdx] else 0.0
        return if (value <= threshold) left else right
    }
}

private class CommonMainCodecModel(
    override val slotId: Int,
    override val slotName: String,
) : CodecModel {
    private val cvdHistory = DoubleArray(64)
    private var cvdCount = 0

    private val lmHighs = DoubleArray(32)
    private val lmLows = DoubleArray(32)
    private val lmCloses = DoubleArray(32)
    private var lmCount = 0

    private var kalmanXPrice = 0.0
    private var kalmanXVelocity = 0.0
    private var p00 = 1.0
    private var p01 = 0.0
    private var p10 = 0.0
    private var p11 = 1.0

    private val forest: Array<DecisionTree> = Array(20) { i -> RF_BASE_TREES[i % RF_BASE_TREES.size] }
    private val stumps: Array<WeakLearner> = XGB_STUMPS

    override fun evaluate(input: CodecInput): CodecSignal {
        return when (slotId) {
            1 -> evalVolatilityBreakout(input)
            2 -> evalMomentumTrend(input)
            3 -> evalMeanReversion(input)
            4 -> evalTrendFollowing(input)
            5 -> evalPairsTrading(input)
            6 -> evalGridTrading(input)
            7 -> evalVolumeProfile(input)
            8 -> evalOrderFlow(input)
            9 -> evalCorrelationTrading(input)
            10 -> evalLiquidityMaking(input)
            11 -> evalSectorRotation(input)
            12 -> evalCompositeAlpha(input)
            13 -> evalRsiReversal(input)
            14 -> evalBollingerBands(input)
            15 -> evalMacdCrossover(input)
            16 -> evalStochasticKd(input)
            17 -> evalAdxTrendStrength(input)
            18 -> evalVwapMeanReversion(input)
            19 -> evalKalmanFilterTrend(input)
            20 -> evalHurstRegime(input)
            21 -> evalRandomForestClassifier(input)
            22 -> evalXgboostSignal(input)
            23 -> evalTransformerAttention(input)
            24 -> evalZscoreStatArb(input)
            else -> signal(
                input = input,
                confidence = 0.1,
                direction = 0.0,
                instruments = mapOf(
                    "gate_reason_code" to GateReason.UNKNOWN_CODEC,
                    "error_unknown_slot_id" to slotId.toDouble(),
                ),
            )
        }
    }

    override fun reset() {
        cvdHistory.fill(0.0)
        cvdCount = 0

        lmHighs.fill(0.0)
        lmLows.fill(0.0)
        lmCloses.fill(0.0)
        lmCount = 0

        kalmanXPrice = 0.0
        kalmanXVelocity = 0.0
        p00 = 1.0
        p01 = 0.0
        p10 = 0.0
        p11 = 1.0
    }

    private fun signal(confidence: Double, direction: Double, instruments: Map<String, Double>): CodecSignal =
        CodecSignal(
            slotId = slotId,
            slotName = slotName,
            confidence = CodecAutoVec.clip01(confidence),
            direction = CodecAutoVec.clip11(direction),
            instruments = instruments,
        )

    private fun signal(
        input: CodecInput,
        confidence: Double,
        direction: Double,
        instruments: Map<String, Double>,
    ): CodecSignal {
        val merged = LinkedHashMap<String, Double>(instruments.size + 16)
        merged["feature_count"] = input.features.size.toDouble()
        merged["market_key_count"] = input.market.size.toDouble()
        merged["close_count"] = (input.closes?.size ?: 0).toDouble()
        merged["high_count"] = (input.highs?.size ?: 0).toDouble()
        merged["low_count"] = (input.lows?.size ?: 0).toDouble()
        merged["volume_count"] = (input.volumes?.size ?: 0).toDouble()
        merged["confidence_raw"] = confidence
        merged["direction_raw"] = direction
        merged["confidence_clamped"] = CodecAutoVec.clip01(confidence)
        merged["direction_clamped"] = CodecAutoVec.clip11(direction)
        if (!instruments.containsKey("gate_reason_code")) {
            merged["gate_reason_code"] = GateReason.ACTIVE
        }
        merged.putAll(instruments)
        return signal(confidence = confidence, direction = direction, instruments = merged)
    }

    private fun gateSignal(
        input: CodecInput,
        gateReasonCode: Double,
        confidence: Double,
        direction: Double = 0.0,
        extras: Map<String, Double> = emptyMap(),
    ): CodecSignal {
        val merged = LinkedHashMap<String, Double>(extras.size + 4)
        merged["gate_reason_code"] = gateReasonCode
        merged["is_gated_path"] = 1.0
        merged.putAll(extras)
        return signal(input = input, confidence = confidence, direction = direction, instruments = merged)
    }

    private fun featureWindow(features: DoubleArray, window: Int = 64): DoubleArray {
        val n = min(features.size, window)
        val out = DoubleArray(n)
        var i = 0
        while (i < n) {
            out[i] = features[i]
            i += 1
        }
        return out
    }

    private fun appendHistory(history: DoubleArray, count: Int, value: Double): Int {
        if (count < history.size) {
            history[count] = value
            return count + 1
        }
        var i = 1
        while (i < history.size) {
            history[i - 1] = history[i]
            i += 1
        }
        history[history.lastIndex] = value
        return history.size
    }

    private fun evalVolatilityBreakout(input: CodecInput): CodecSignal {
        val price = CodecAutoVec.market(input, "price", 0.0)
        val high = CodecAutoVec.market(input, "high", price)
        val low = CodecAutoVec.market(input, "low", price)
        val atr = CodecAutoVec.market(input, "atr_14", (high - low) * 0.5)
        val momentum = CodecAutoVec.market(input, "momentum", 0.0)

        val rangeExpansion = CodecAutoVec.safeDiv(high - low, atr + 1e-8, 0.0)
        val volatilitySignal = if (rangeExpansion > 2.0) {
            CodecAutoVec.sign(momentum) * min(rangeExpansion / 3.0, 1.0)
        } else {
            0.0
        }

        return signal(input, 
            confidence = abs(volatilitySignal) + 0.3,
            direction = volatilitySignal,
            instruments = mapOf(
                "volatility_signal" to volatilitySignal,
                "atr_norm" to CodecAutoVec.safeDiv(atr, abs(price) + 1e-8, 0.0),
                "momentum" to momentum,
            ),
        )
    }

    private fun evalMomentumTrend(input: CodecInput): CodecSignal {
        val price = CodecAutoVec.market(input, "price", 0.0)
        val ema10 = CodecAutoVec.market(input, "ema_5", price)
        val ema30 = CodecAutoVec.market(input, "ema_15", price)
        val ema60 = CodecAutoVec.market(input, "ema_60", price)
        val momentum = CodecAutoVec.market(input, "momentum", 0.0)

        var trendAlignment = 0
        trendAlignment += if (ema10 > ema30) 1 else -1
        trendAlignment += if (ema30 > ema60) 1 else -1
        val trendSignal = (trendAlignment / 2.0) * 0.7 + CodecAutoVec.sign(momentum) * 0.3

        val features = featureWindow(input.features)
        return signal(input, 
            confidence = abs(trendSignal) + 0.25,
            direction = trendSignal,
            instruments = mapOf(
                "momentum_fast" to momentum,
                "returns_last" to if (features.isNotEmpty()) features[0] else 0.0,
            ),
        )
    }

    private fun evalMeanReversion(input: CodecInput): CodecSignal {
        val price = CodecAutoVec.market(input, "price", 0.0)
        val sma20 = CodecAutoVec.market(input, "sma_15", price)
        val rollingStd = CodecAutoVec.market(input, "vol_5m", abs(price) * 0.02)

        val zScore = CodecAutoVec.safeDiv(price - sma20, rollingStd + 1e-8, 0.0)
        val reversionSignal = if (abs(zScore) > 2.0) {
            -CodecAutoVec.sign(zScore) * min(abs(zScore) / 4.0, 1.0)
        } else {
            0.0
        }

        val rsi = CodecAutoVec.market(input, "rsi_14", 50.0)
        val rsiSignal = when {
            rsi > 70.0 -> -0.5
            rsi < 30.0 -> 0.5
            else -> 0.0
        }

        val combined = reversionSignal * 0.6 + rsiSignal * 0.4
        val features = featureWindow(input.features)

        return signal(input, 
            confidence = abs(combined) + 0.2,
            direction = combined,
            instruments = mapOf(
                "returns_last" to if (features.isNotEmpty()) features[0] else 0.0,
            ),
        )
    }

    private fun evalTrendFollowing(input: CodecInput): CodecSignal {
        val price = CodecAutoVec.market(input, "price", 0.0)
        val sma5 = CodecAutoVec.market(input, "sma_5", price)
        val sma15 = CodecAutoVec.market(input, "sma_15", price)
        val sma60 = CodecAutoVec.market(input, "sma_60", price)

        var trendScore = 0.0
        trendScore += if (sma5 > sma15) 1.0 else -1.0
        trendScore += if (sma15 > sma60) 1.0 else -1.0
        trendScore += if (price > sma5) 0.5 else -0.5

        val adx = CodecAutoVec.market(input, "adx_14", 0.0)
        val trendStrength = if (adx > 0.0) adx / 100.0 else 0.0

        return signal(input, 
            confidence = min(1.0, trendStrength + 0.3),
            direction = trendScore / 2.5,
            instruments = mapOf(
                "momentum" to CodecAutoVec.market(input, "momentum", 0.0),
            ),
        )
    }

    private fun evalPairsTrading(input: CodecInput): CodecSignal {
        val returns = featureWindow(input.features)
        if (returns.size < 30) {
            return gateSignal(
                input = input,
                gateReasonCode = GateReason.INSUFFICIENT_FEATURE_HISTORY,
                confidence = 0.1,
                extras = mapOf(
                    "required_feature_count" to 30.0,
                    "observed_feature_count" to returns.size.toDouble(),
                ),
            )
        }

        val ohlcv = CodecAutoVec.resolveOhlcv(input)
        val prices = ohlcv.closes
        val n = prices.size
        if (n < 30) {
            return gateSignal(
                input = input,
                gateReasonCode = GateReason.INSUFFICIENT_OHLCV_HISTORY,
                confidence = 0.1,
                extras = mapOf(
                    "required_close_count" to 30.0,
                    "observed_close_count" to n.toDouble(),
                ),
            )
        }

        val legA = DoubleArray(n)
        val legB = DoubleArray(n)
        legA[0] = prices[0]
        legB[0] = prices[0]
        var i = 1
        while (i < n) {
            legA[i] = 0.05 * prices[i] + 0.95 * legA[i - 1]
            legB[i] = 0.15 * prices[i] + 0.85 * legB[i - 1]
            i += 1
        }

        val spread = DoubleArray(n)
        i = 0
        while (i < n) {
            spread[i] = legB[i] - legA[i]
            i += 1
        }
        val start = CodecAutoVec.tailStart(n, 30)
        val spreadMu = CodecAutoVec.mean(spread, start, n)
        val spreadStd = CodecAutoVec.std(spread, start, n)
        val z = CodecAutoVec.safeDiv(spread[n - 1] - spreadMu, spreadStd + 1e-8, 0.0)

        var cov = 0.0
        var varLag = 0.0
        var pairs = 0
        i = start + 1
        while (i < n) {
            val delta = spread[i] - spread[i - 1]
            val yLag = spread[i - 1]
            cov += delta * yLag
            varLag += yLag * yLag
            pairs += 1
            i += 1
        }
        val beta = if (pairs > 0 && varLag > 1e-8) cov / varLag else 0.0
        val reverting = beta < -0.05

        val (direction, confidence) = if (abs(z) < 1.5 || !reverting) {
            0.0 to 0.15
        } else {
            val dir = -CodecAutoVec.sign(z)
            val halfLife = if (beta < 0.0) -ln(2.0) / (beta + 1e-10) else 999.0
            val hlFactor = min(1.0, 20.0 / (halfLife + 1.0))
            val conf = min(1.0, (abs(z) - 1.5) / 2.0 * hlFactor + 0.3)
            dir to conf
        }

        return signal(input, 
            confidence = confidence,
            direction = direction,
            instruments = mapOf(
                "spread_z" to z,
                "ar1_beta" to beta,
            ),
        )
    }

    private fun evalGridTrading(input: CodecInput): CodecSignal {
        val p = GRID_TRADING_PARAMS
        val price = CodecAutoVec.market(input, "price", 1.0)
        val high = CodecAutoVec.market(input, "high", price)
        val low = CodecAutoVec.market(input, "low", price)
        val atr = CodecAutoVec.market(input, "atr_14", high - low)
        val returns = featureWindow(input.features)
        if (returns.size < p.window) {
            return gateSignal(
                input = input,
                gateReasonCode = GateReason.INSUFFICIENT_FEATURE_HISTORY,
                confidence = 0.15,
                extras = mapOf(
                    "required_window" to p.window.toDouble(),
                    "observed_feature_count" to returns.size.toDouble(),
                ),
            )
        }

        val closes = CodecAutoVec.resolveOhlcv(input).closes
        val start = CodecAutoVec.tailStart(closes.size, p.window)
        val midline = CodecAutoVec.mean(closes, start, closes.size)
        val vol = CodecAutoVec.std(closes, start, closes.size)
        val atrComponent = atr * p.atrBandMultiplier
        val sigmaComponent = vol * p.sigmaBandMultiplier
        val floorComponent = abs(price) * p.minBandFloorPct
        val bandHalf = max(max(atrComponent, sigmaComponent), floorComponent)
        val lower = midline - bandHalf
        val upper = midline + bandHalf
        val inBand = price > lower && price < upper
        if (!inBand) {
            return gateSignal(
                input = input,
                gateReasonCode = GateReason.OUTSIDE_DYNAMIC_BAND,
                confidence = 0.1,
                extras = mapOf(
                    "price" to price,
                    "midline" to midline,
                    "band_half" to bandHalf,
                    "band_lower" to lower,
                    "band_upper" to upper,
                    "atr_component" to atrComponent,
                    "sigma_component" to sigmaComponent,
                    "floor_component" to floorComponent,
                ),
            )
        }

        val bandPos = 2.0 * CodecAutoVec.safeDiv(price - lower, (upper - lower) + 1e-8, 0.5) - 1.0
        val levelStep = 2.0 / p.gridLevels.toDouble()
        val halfLevel = levelStep / 2.0
        val gridSignal = -bandPos
        // Python-style positive modulo semantics for negative band positions.
        val edgePhase = ((bandPos % halfLevel) + halfLevel) % halfLevel
        val edgeProximity = 1.0 - abs(edgePhase)
        val confidenceBeforeRegime = min(
            1.0,
            p.baseConfidence +
                abs(gridSignal) * p.gridSignalWeight +
                edgeProximity * p.edgeWeight,
        )
        var confidence = confidenceBeforeRegime

        val pStart = CodecAutoVec.tailStart(closes.size, p.window)
        var signChanges = 0
        var prevDiffSign = 0.0
        var i = pStart + 1
        while (i < closes.size) {
            val diffSign = CodecAutoVec.sign(closes[i] - closes[i - 1])
            if (prevDiffSign != 0.0 && diffSign != 0.0 && diffSign != prevDiffSign) signChanges += 1
            if (diffSign != 0.0) prevDiffSign = diffSign
            i += 1
        }
        val denom = max(1, (closes.size - pStart - 1))
        val oscillation = signChanges.toDouble() / denom.toDouble()
        val regimeMultiplier = p.oscillationBase + oscillation * p.oscillationSlope
        confidence *= regimeMultiplier
        val direction = CodecAutoVec.tanhStable(gridSignal * 2.0)

        return signal(input, 
            confidence = confidence,
            direction = direction,
            instruments = mapOf(
                "gate_reason_code" to GateReason.ACTIVE,
                "price" to price,
                "midline" to midline,
                "band_half" to bandHalf,
                "band_lower" to lower,
                "band_upper" to upper,
                "atr_component" to atrComponent,
                "sigma_component" to sigmaComponent,
                "floor_component" to floorComponent,
                "band_pos" to bandPos,
                "grid_signal" to gridSignal,
                "edge_phase" to edgePhase,
                "edge_proximity" to edgeProximity,
                "confidence_pre_regime" to confidenceBeforeRegime,
                "regime_multiplier" to regimeMultiplier,
                "confidence_post_regime" to confidence,
                "direction_tanh" to direction,
                "sign_change_count" to signChanges.toDouble(),
                "sign_change_denom" to denom.toDouble(),
                "oscillation" to oscillation,
            ),
        )
    }

    private fun evalVolumeProfile(input: CodecInput): CodecSignal {
        val price = CodecAutoVec.market(input, "price", 1.0)
        val returns = featureWindow(input.features)
        if (returns.size < 10) {
            return gateSignal(
                input = input,
                gateReasonCode = GateReason.INSUFFICIENT_FEATURE_HISTORY,
                confidence = 0.15,
                extras = mapOf(
                    "required_feature_count" to 10.0,
                    "observed_feature_count" to returns.size.toDouble(),
                ),
            )
        }

        val closes = CodecAutoVec.resolveOhlcv(input).closes
        val w = min(closes.size, min(returns.size, 50))
        if (w < 3) {
            val gateReason = if (closes.size < 3) GateReason.INSUFFICIENT_OHLCV_HISTORY else GateReason.INSUFFICIENT_FEATURE_HISTORY
            return gateSignal(
                input = input,
                gateReasonCode = gateReason,
                confidence = 0.15,
                extras = mapOf(
                    "required_window" to 3.0,
                    "observed_window" to w.toDouble(),
                    "observed_feature_count" to returns.size.toDouble(),
                    "observed_close_count" to closes.size.toDouble(),
                ),
            )
        }

        val pStart = closes.size - w
        val rStart = returns.size - w
        var pMin = closes[pStart]
        var pMax = closes[pStart]
        var i = pStart
        while (i < closes.size) {
            val p = closes[i]
            if (p < pMin) pMin = p
            if (p > pMax) pMax = p
            i += 1
        }
        val range = (pMax - pMin) + 1e-8
        if (range <= 1e-8) {
            return gateSignal(
                input = input,
                gateReasonCode = GateReason.DEGENERATE_RANGE,
                confidence = 0.15,
                extras = mapOf(
                    "price_min" to pMin,
                    "price_max" to pMax,
                    "price_range" to range,
                ),
            )
        }

        val bins = 20
        val volHist = DoubleArray(bins)
        var volSum = 0.0
        i = rStart
        while (i < returns.size) {
            volSum += abs(returns[i]) + 1e-6
            i += 1
        }
        i = 0
        while (i < w) {
            val p = closes[pStart + i]
            val volProxy = (abs(returns[rStart + i]) + 1e-6) / (volSum + 1e-8)
            val rawBin = ((p - pMin) / range * bins.toDouble()).toInt()
            val bin = rawBin.coerceIn(0, bins - 1)
            volHist[bin] += volProxy
            i += 1
        }

        var vpocBin = 0
        var vpocVal = volHist[0]
        i = 1
        while (i < bins) {
            if (volHist[i] > vpocVal) {
                vpocVal = volHist[i]
                vpocBin = i
            }
            i += 1
        }
        val binSize = range / bins.toDouble()
        val vpocPrice = pMin + (vpocBin + 0.5) * binSize
        val pStd = CodecAutoVec.std(closes, pStart, closes.size) + 1e-8
        val distZ = (price - vpocPrice) / pStd
        val direction = if (abs(distZ) > 0.3) -CodecAutoVec.sign(distZ) else 0.0

        val currentBin = (((price - pMin) / range) * bins.toDouble()).toInt().coerceIn(0, bins - 1)
        var volAbove = 0.0
        var volBelow = 0.0
        i = 0
        while (i < bins) {
            if (i < currentBin) volBelow += volHist[i]
            if (i > currentBin) volAbove += volHist[i]
            i += 1
        }
        val volSkew = volBelow - volAbove
        val confidence = min(1.0, abs(distZ) * 0.4 + abs(volSkew) * 0.4 + 0.2)

        return signal(input, 
            confidence = confidence,
            direction = direction,
            instruments = mapOf(
                "vpoc_dist_z" to distZ,
                "vol_skew" to volSkew,
            ),
        )
    }

    private fun evalOrderFlow(input: CodecInput): CodecSignal {
        val price = CodecAutoVec.market(input, "price", 1.0)
        val high = CodecAutoVec.market(input, "high", price)
        val low = CodecAutoVec.market(input, "low", price)
        val volume = CodecAutoVec.market(input, "volume", 1.0)
        val atr = CodecAutoVec.market(input, "atr_14", (high - low) + 1e-8)

        val returns = featureWindow(input.features)
        val openProxy = if (returns.size >= 2) {
            price / (1.0 + returns[returns.lastIndex] + 1e-8)
        } else {
            (high + low) * 0.5
        }
        val barRange = CodecAutoVec.barRange(high, low)
        val netMove = price - openProxy
        val barDelta = netMove / barRange
        val relVol = min(volume / (atr * 1e4 + 1e-8), 3.0)
        val volDelta = barDelta * relVol

        cvdCount = appendHistory(cvdHistory, cvdCount, volDelta)
        val cvdStart = CodecAutoVec.tailStart(cvdCount, 20)
        val cvd = CodecAutoVec.sum(cvdHistory, cvdStart, cvdCount)
        val cvdMean = CodecAutoVec.mean(cvdHistory, cvdStart, cvdCount)
        val cvdStd = CodecAutoVec.std(cvdHistory, cvdStart, cvdCount) + 1e-8
        val cvdZ = (cvd - cvdMean * (cvdCount - cvdStart).toDouble()) /
            (cvdStd * sqrt((cvdCount - cvdStart).toDouble()) + 1e-8)

        val absorption = ((barRange / (atr + 1e-8)) > 1.5) && (abs(barDelta) < 0.15)
        val (direction, confidence) = if (absorption) {
            val dir = if (netMove != 0.0) -CodecAutoVec.sign(netMove) else 0.0
            val conf = min(1.0, 0.6 + abs(barDelta) * 0.2)
            dir to conf
        } else {
            val dir = if (abs(cvdZ) > 0.5) CodecAutoVec.sign(cvdZ) else 0.0
            val conf = min(1.0, 0.2 + abs(cvdZ) * 0.2 + abs(barDelta) * 0.3)
            dir to conf
        }

        return signal(input, 
            confidence = confidence,
            direction = direction,
            instruments = mapOf(
                "bar_delta" to barDelta,
                "cvd_z" to cvdZ,
                "absorption" to if (absorption) 1.0 else 0.0,
            ),
        )
    }

    private fun evalCorrelationTrading(input: CodecInput): CodecSignal {
        val returns = featureWindow(input.features)
        if (returns.size < 40) {
            return gateSignal(
                input = input,
                gateReasonCode = GateReason.INSUFFICIENT_FEATURE_HISTORY,
                confidence = 0.15,
                extras = mapOf(
                    "required_feature_count" to 40.0,
                    "observed_feature_count" to returns.size.toDouble(),
                ),
            )
        }

        val lookbacks = intArrayOf(5, 10, 20, 40)
        val signals = DoubleArray(lookbacks.size)
        var signalCount = 0
        var acfAccumulator = 0.0
        var lbCount = 0

        var i = 0
        while (i < lookbacks.size) {
            val lb = lookbacks[i]
            if (returns.size >= lb) {
                val start = returns.size - lb
                val weightedAcf = weightedAutocorr(returns, start, returns.size, min(5, lb / 2))
                val recentMean = CodecAutoVec.mean(returns, max(start, returns.size - 3), returns.size)
                val recentDir = CodecAutoVec.sign(recentMean)
                if (abs(weightedAcf) > 0.05) {
                    signals[signalCount] = CodecAutoVec.sign(weightedAcf) * recentDir
                    signalCount += 1
                }
                acfAccumulator += weightedAcf
                lbCount += 1
            }
            i += 1
        }

        if (signalCount == 0) {
            return signal(input, 
                confidence = 0.15,
                direction = 0.0,
                instruments = mapOf("weighted_acf" to 0.0, "breadth" to 0.0),
            )
        }

        var dirSum = 0.0
        var signSum = 0.0
        i = 0
        while (i < signalCount) {
            dirSum += signals[i]
            signSum += CodecAutoVec.sign(signals[i])
            i += 1
        }
        val direction = CodecAutoVec.clip11(dirSum / signalCount.toDouble())
        val breadth = abs(signSum / signalCount.toDouble())
        val confidence = min(1.0, breadth * 0.6 + abs(direction) * 0.2 + 0.15)

        return signal(input, 
            confidence = confidence,
            direction = direction,
            instruments = mapOf(
                "weighted_acf" to (if (lbCount > 0) acfAccumulator / lbCount.toDouble() else 0.0),
                "breadth" to breadth,
            ),
        )
    }

    private fun evalLiquidityMaking(input: CodecInput): CodecSignal {
        val price = CodecAutoVec.market(input, "price", 1.0)
        val high = CodecAutoVec.market(input, "high", price)
        val low = CodecAutoVec.market(input, "low", price)
        val returns = featureWindow(input.features)

        lmCount = appendHistory(lmHighs, lmCount, high)
        val lowCount = appendHistory(lmLows, if (lmCount == 0) 0 else lmCount - 1, low)
        lmCount = max(lmCount, lowCount)
        val closeCount = appendHistory(lmCloses, if (lmCount == 0) 0 else lmCount - 1, price)
        lmCount = max(lmCount, closeCount)

        if (returns.size < 4 || lmCount < 4) {
            val gateReason = if (returns.size < 4) GateReason.INSUFFICIENT_FEATURE_HISTORY else GateReason.INSUFFICIENT_OHLCV_HISTORY
            return gateSignal(
                input = input,
                gateReasonCode = gateReason,
                confidence = 0.15,
                extras = mapOf(
                    "required_feature_count" to 4.0,
                    "required_liquidity_history" to 4.0,
                    "observed_feature_count" to returns.size.toDouble(),
                    "observed_liquidity_history" to lmCount.toDouble(),
                ),
            )
        }
        val start = CodecAutoVec.tailStart(returns.size, 20)
        val cov = CodecAutoVec.covarianceLag1(returns, start, returns.size)
        val rollSpread = if (cov < 0.0) 2.0 * sqrt(-cov) else 0.0
        val bouncing = cov < -1e-6

        val gkVol = garmanKlassVol()
        val recentRet = returns[returns.lastIndex]

        var direction: Double
        var confidence: Double
        if (bouncing) {
            direction = if (recentRet != 0.0) -CodecAutoVec.sign(recentRet) else 0.0
            confidence = min(1.0, rollSpread * 20.0 + 0.3)
        } else {
            direction = if (abs(recentRet) > gkVol) CodecAutoVec.sign(recentRet) else 0.0
            confidence = min(1.0, abs(recentRet) / (gkVol + 1e-8) * 0.3 + 0.2)
        }
        val volPenalty = min(0.4, gkVol * 20.0)
        confidence = max(0.1, confidence - volPenalty)

        return signal(input, 
            confidence = confidence,
            direction = direction,
            instruments = mapOf(
                "roll_spread" to rollSpread,
                "gk_vol" to gkVol,
                "bouncing" to if (bouncing) 1.0 else 0.0,
            ),
        )
    }

    private fun evalSectorRotation(input: CodecInput): CodecSignal {
        val returns = featureWindow(input.features)
        if (returns.size < 40) {
            return gateSignal(
                input = input,
                gateReasonCode = GateReason.INSUFFICIENT_FEATURE_HISTORY,
                confidence = 0.15,
                extras = mapOf(
                    "required_feature_count" to 40.0,
                    "observed_feature_count" to returns.size.toDouble(),
                ),
            )
        }
        val closes = CodecAutoVec.resolveOhlcv(input).closes
        if (closes.size < 40) {
            return gateSignal(
                input = input,
                gateReasonCode = GateReason.INSUFFICIENT_OHLCV_HISTORY,
                confidence = 0.15,
                extras = mapOf(
                    "required_close_count" to 40.0,
                    "observed_close_count" to closes.size.toDouble(),
                ),
            )
        }

        val windows = intArrayOf(5, 10, 20, 40)
        val weights = doubleArrayOf(0.40, 0.30, 0.20, 0.10)
        val momentumScores = DoubleArray(windows.size)
        var i = 0
        while (i < windows.size) {
            val w = windows[i]
            if (closes.size > w) {
                val base = closes[closes.size - w]
                momentumScores[i] = CodecAutoVec.safeDiv(closes[closes.lastIndex] - base, base + 1e-8, 0.0)
            }
            i += 1
        }
        val composite = CodecAutoVec.dot(momentumScores, weights)

        val ret1 = returns[returns.lastIndex]
        val histStart = CodecAutoVec.tailStart(returns.size, 60)
        val hist = DoubleArray(returns.size - histStart)
        i = 0
        while (i < hist.size) {
            hist[i] = returns[histStart + i]
            i += 1
        }
        val pct = CodecAutoVec.percentileRank(hist, ret1)

        val slopeStart = CodecAutoVec.tailStart(closes.size, 20)
        val regime = CodecAutoVec.sign(CodecAutoVec.linearSlope(closes, slopeStart, closes.size))
        var direction = CodecAutoVec.sign(composite) * (0.7 + 0.3 * abs(2.0 * pct - 1.0))
        if (regime != 0.0) direction = direction * 0.7 + regime * 0.3
        val confidence = min(1.0, abs(composite) * 10.0 + 0.25 + abs(2.0 * pct - 1.0) * 0.2)

        return signal(input, 
            confidence = confidence,
            direction = direction,
            instruments = mapOf(
                "composite_mom" to composite,
                "pct_rank" to pct,
                "regime" to regime,
            ),
        )
    }

    private fun evalCompositeAlpha(input: CodecInput): CodecSignal {
        val returns = featureWindow(input.features)
        if (returns.size < 5) {
            return gateSignal(
                input = input,
                gateReasonCode = GateReason.INSUFFICIENT_FEATURE_HISTORY,
                confidence = 0.2,
                extras = mapOf(
                    "required_feature_count" to 5.0,
                    "observed_feature_count" to returns.size.toDouble(),
                ),
            )
        }

        val price = CodecAutoVec.market(input, "price", 1.0)
        val volume = CodecAutoVec.market(input, "volume", 1.0)
        val prices = CodecAutoVec.resolveOhlcv(input).closes

        val alphaSignals = DoubleArray(5)
        val alphaQuality = DoubleArray(5)

        var idx = 0
        val momentum = alphaMomentum(prices)
        alphaSignals[idx] = momentum.first
        alphaQuality[idx] = momentum.second
        idx += 1

        val rsi = alphaRsi(returns)
        alphaSignals[idx] = rsi.first
        alphaQuality[idx] = rsi.second
        idx += 1

        val bollinger = alphaBollinger(prices)
        alphaSignals[idx] = bollinger.first
        alphaQuality[idx] = bollinger.second
        idx += 1

        val donchian = alphaDonchian(prices)
        alphaSignals[idx] = donchian.first
        alphaQuality[idx] = donchian.second
        idx += 1

        val volMom = alphaVolumeMomentum(returns, price, volume)
        alphaSignals[idx] = volMom.first
        alphaQuality[idx] = volMom.second

        val totalQuality = CodecAutoVec.sum(alphaQuality)
        if (totalQuality < 1e-8) {
            return gateSignal(
                input = input,
                gateReasonCode = GateReason.NO_ALPHA_QUALITY,
                confidence = 0.2,
                extras = mapOf(
                    "alpha_quality_sum" to totalQuality,
                    "alpha_signal_count" to alphaSignals.size.toDouble(),
                ),
            )
        }

        var weighted = 0.0
        var nonZero = 0
        var signSum = 0.0
        idx = 0
        while (idx < alphaSignals.size) {
            weighted += alphaSignals[idx] * alphaQuality[idx]
            if (alphaSignals[idx] != 0.0) {
                nonZero += 1
                signSum += CodecAutoVec.sign(alphaSignals[idx])
            }
            idx += 1
        }
        val direction = weighted / totalQuality
        val agreement = if (nonZero > 0) abs(signSum / nonZero.toDouble()) else 0.0
        val confidence = min(1.0, agreement * 0.5 + abs(direction) * 0.3 + 0.2)

        return signal(input, 
            confidence = confidence,
            direction = direction,
            instruments = mapOf(
                "alpha_count" to nonZero.toDouble(),
                "alpha_agreement" to agreement,
            ),
        )
    }

    private fun evalRsiReversal(input: CodecInput): CodecSignal {
        val oversold = 30.0
        val overbought = 70.0
        val rsi = CodecAutoVec.market(input, "rsi_14", 50.0)
        val stoch = CodecAutoVec.market(input, "stochastic", 50.0)

        var direction = 0.0
        var confidence = 0.2
        if (rsi < oversold) {
            direction = (oversold - rsi) / oversold
            confidence = min(1.0, 1.0 - rsi / oversold + 0.3)
        } else if (rsi > overbought) {
            direction = -(rsi - overbought) / (100.0 - overbought)
            confidence = min(1.0, (rsi - overbought) / (100.0 - overbought) + 0.3)
        }

        if (stoch < 20.0 && rsi < oversold) {
            direction *= 1.3
            confidence = min(1.0, confidence * 1.2)
        } else if (stoch > 80.0 && rsi > overbought) {
            direction *= 1.3
            confidence = min(1.0, confidence * 1.2)
        }

        return signal(input, 
            confidence = confidence,
            direction = direction,
            instruments = mapOf("rsi_14" to rsi),
        )
    }

    private fun evalBollingerBands(input: CodecInput): CodecSignal {
        val price = CodecAutoVec.market(input, "price", 0.0)
        val bbUpper = CodecAutoVec.market(input, "bb_upper", price)
        val bbLower = CodecAutoVec.market(input, "bb_lower", price)
        val sma20 = CodecAutoVec.market(input, "sma_15", price)

        var direction = 0.0
        var confidence = 0.2
        var bbPosition = 0.5

        if (bbUpper > bbLower) {
            val bbWidth = bbUpper - bbLower
            bbPosition = CodecAutoVec.safeDiv(price - bbLower, bbWidth + 1e-8, 0.5)
            if (bbPosition < 0.0) {
                direction = 0.8
                confidence = min(1.0, abs(bbPosition) + 0.4)
            } else if (bbPosition > 1.0) {
                direction = -0.8
                confidence = min(1.0, bbPosition - 1.0 + 0.4)
            } else if (bbPosition < 0.2) {
                direction = 0.5
                confidence = 0.5
            } else if (bbPosition > 0.8) {
                direction = -0.5
                confidence = 0.5
            }
        }

        val bandwidth = if (sma20 != 0.0) (bbUpper - bbLower) / (sma20 + 1e-8) else 0.0
        if (bandwidth < 0.02) confidence *= 0.5

        return signal(input, 
            confidence = confidence,
            direction = direction,
            instruments = mapOf(
                "bb_pct" to bbPosition,
                "bb_width" to bandwidth,
            ),
        )
    }

    private fun evalMacdCrossover(input: CodecInput): CodecSignal {
        val macd = CodecAutoVec.market(input, "macd", 0.0)
        val macdSignal = CodecAutoVec.market(input, "macd_signal", 0.0)
        val macdHist = CodecAutoVec.market(input, "macd_hist", macd - macdSignal)
        val momentum = CodecAutoVec.market(input, "momentum", 0.0)

        var direction = 0.0
        var confidence = 0.2
        if (macdHist > 0.0) {
            direction = min(macdHist * 20.0, 1.0)
            confidence = min(1.0, abs(macdHist) * 10.0 + 0.3)
        } else if (macdHist < 0.0) {
            direction = max(macdHist * 20.0, -1.0)
            confidence = min(1.0, abs(macdHist) * 10.0 + 0.3)
        }
        if (CodecAutoVec.sign(direction) == CodecAutoVec.sign(momentum) && direction != 0.0) {
            confidence *= 1.2
        }

        return signal(input, 
            confidence = confidence,
            direction = direction,
            instruments = mapOf(
                "macd_hist" to macdHist,
                "macd_line" to macd,
            ),
        )
    }

    private fun evalStochasticKd(input: CodecInput): CodecSignal {
        val oversold = 20.0
        val overbought = 80.0
        val stochK = CodecAutoVec.market(input, "stoch_k", 50.0)
        val stochD = CodecAutoVec.market(input, "stoch_d", 50.0)
        val cross = stochK - stochD

        var direction: Double
        var confidence: Double
        if (stochK < oversold && stochD < oversold) {
            if (cross > 0.0) {
                direction = min(cross / 10.0 + 0.5, 1.0)
                confidence = 0.7 + (oversold - stochK) / 100.0
            } else {
                direction = 0.0
                confidence = 0.2
            }
        } else if (stochK > overbought && stochD > overbought) {
            if (cross < 0.0) {
                direction = max(cross / 10.0 - 0.5, -1.0)
                confidence = 0.7 + (stochK - overbought) / 100.0
            } else {
                direction = 0.0
                confidence = 0.2
            }
        } else {
            direction = CodecAutoVec.sign(cross) * min(abs(cross) / 10.0, 0.5)
            confidence = 0.4
        }

        return signal(input, 
            confidence = confidence,
            direction = direction,
            instruments = mapOf(
                "stoch_k" to stochK,
                "stoch_d" to stochD,
            ),
        )
    }

    private fun evalAdxTrendStrength(input: CodecInput): CodecSignal {
        val adx = CodecAutoVec.market(input, "adx", CodecAutoVec.market(input, "adx_14", 0.0))
        val plusDi = CodecAutoVec.market(input, "plus_di", 0.0)
        val minusDi = CodecAutoVec.market(input, "minus_di", 0.0)
        val momentum = CodecAutoVec.market(input, "momentum", CodecAutoVec.market(input, "log_return", 0.0))

        val diDiff = plusDi - minusDi
        var direction = CodecAutoVec.tanhStable(diDiff / 20.0)
        val adxFactor = when {
            adx >= 25.0 -> min(1.0, adx / 50.0 + 0.3)
            adx >= 15.0 -> 0.15 + (adx - 15.0) / 10.0 * 0.3
            else -> 0.1 + adx / 15.0 * 0.1
        }
        var confidence = min(1.0, adxFactor + abs(diDiff) / 50.0)

        if (CodecAutoVec.sign(direction) == CodecAutoVec.sign(momentum) && adx > 20.0) {
            confidence = min(1.0, confidence * 1.2)
        }

        return signal(input, 
            confidence = confidence,
            direction = direction,
            instruments = mapOf(
                "adx" to adx,
                "plus_di" to plusDi,
                "minus_di" to minusDi,
            ),
        )
    }

    private fun evalVwapMeanReversion(input: CodecInput): CodecSignal {
        val price = CodecAutoVec.market(input, "price", 0.0)
        val vwap = CodecAutoVec.market(input, "vwap", price)
        val volume = CodecAutoVec.market(input, "volume", 0.0)
        val avgVolume = CodecAutoVec.market(input, "avg_volume", if (volume == 0.0) 1.0 else volume)
        val regime = CodecAutoVec.market(input, "regime_label", 1.0)

        var deviation = 0.0
        var direction = 0.0
        var confidence = 0.2
        if (vwap > 0.0) {
            deviation = (price - vwap) / vwap
            if (abs(deviation) > 0.01) {
                direction = -CodecAutoVec.sign(deviation) * min(abs(deviation) * 20.0, 1.0)
                confidence = min(1.0, abs(deviation) * 30.0 + 0.3)
            }
            val volRatio = volume / (avgVolume + 1e-8)
            if (volRatio > 1.5) confidence = min(1.0, confidence * 1.2)
        }

        confidence = if (regime == 1.0) confidence * 1.2 else confidence * 0.7

        return signal(input, 
            confidence = confidence,
            direction = direction,
            instruments = mapOf("vwap_dev" to deviation),
        )
    }

    private fun evalKalmanFilterTrend(input: CodecInput): CodecSignal {
        val price = CodecAutoVec.market(input, "price", 0.0)
        val (smoothedPrice, velocity) = kalmanUpdate(price)

        var direction = 0.0
        var confidence = 0.2
        if (abs(velocity) > 0.01) {
            direction = CodecAutoVec.sign(velocity) * min(abs(velocity) * 10.0, 1.0)
            confidence = min(1.0, abs(velocity) * 20.0 + 0.3)
        }

        val innovation = price - smoothedPrice
        if (abs(innovation) > abs(price) * 0.02) {
            direction += -CodecAutoVec.sign(innovation) * 0.2
        }

        return signal(input, 
            confidence = confidence,
            direction = direction,
            instruments = mapOf(
                "kalman_price" to kalmanXPrice,
                "kalman_velocity" to kalmanXVelocity,
            ),
        )
    }

    private fun evalHurstRegime(input: CodecInput): CodecSignal {
        val returns = featureWindow(input.features)
        val price = CodecAutoVec.market(input, "price", 0.0)
        val sma = CodecAutoVec.market(input, "sma_15", price)
        val hurst = estimateHurst(returns)

        var direction = 0.0
        var confidence = 0.2
        if (hurst > 0.55) {
            val trend = price - sma
            direction = CodecAutoVec.sign(trend) * min(abs(trend) / (abs(price) * 0.02 + 1e-8), 1.0)
            confidence = min(1.0, (hurst - 0.5) * 4.0 + 0.3)
        } else if (hurst < 0.45) {
            val deviation = CodecAutoVec.safeDiv(price - sma, sma + 1e-8, 0.0)
            direction = -CodecAutoVec.sign(deviation) * min(abs(deviation) * 10.0, 1.0)
            confidence = min(1.0, (0.5 - hurst) * 4.0 + 0.3)
        } else {
            confidence = 0.3
        }

        return signal(input, 
            confidence = confidence,
            direction = direction,
            instruments = mapOf("hurst_exponent" to hurst),
        )
    }

    private fun evalRandomForestClassifier(input: CodecInput): CodecSignal {
        val engineered = buildRandomForestFeatures(input)
        var voteSum = 0.0
        var i = 0
        while (i < forest.size) {
            voteSum += forest[i].predict(engineered)
            i += 1
        }
        val rawDirection = voteSum / forest.size.toDouble()
        val voteAgreement = abs(rawDirection)
        val direction = if (voteAgreement > 0.1) CodecAutoVec.sign(rawDirection) else 0.0
        val confidence = min(1.0, voteAgreement * 1.2 + 0.2)

        return signal(input, 
            confidence = confidence,
            direction = direction,
            instruments = mapOf(
                "rf_vote_agreement" to voteAgreement,
                "rf_raw_direction" to rawDirection,
            ),
        )
    }

    private fun evalXgboostSignal(input: CodecInput): CodecSignal {
        val engineered = buildXgboostFeatures(input)
        var raw = 0.0
        var i = 0
        while (i < stumps.size) {
            raw += stumps[i].predict(engineered)
            i += 1
        }
        val direction = CodecAutoVec.tanhStable(raw)
        val confidence = min(1.0, abs(direction) + 0.15)

        return signal(input, 
            confidence = confidence,
            direction = direction,
            instruments = mapOf(
                "xgb_raw" to raw,
                "rsi_norm" to if (engineered.size > 4) engineered[4] else 0.5,
                "bb_pos" to if (engineered.size > 6) engineered[6] else 0.0,
            ),
        )
    }

    private fun evalTransformerAttention(input: CodecInput): CodecSignal {
        val features = featureWindow(input.features)
        val tokens = buildAttentionTokens(input)
        val d = tokens[0].size
        val last = tokens.last()
        val scores = DoubleArray(tokens.size)
        var i = 0
        while (i < tokens.size) {
            scores[i] = CodecAutoVec.dot(last, tokens[i]) / sqrt(d.toDouble() + 1e-8)
            i += 1
        }
        val attn = CodecAutoVec.softmax(scores)

        val context = DoubleArray(d)
        i = 0
        while (i < tokens.size) {
            val w = attn[i]
            var j = 0
            while (j < d) {
                context[j] += w * tokens[i][j]
                j += 1
            }
            i += 1
        }

        val directionLogit = CodecAutoVec.dot(context, ATTN_DIR_WEIGHTS, min(context.size, ATTN_DIR_WEIGHTS.size))
        val confLogit = CodecAutoVec.dot(context, ATTN_CONF_WEIGHTS, min(context.size, ATTN_CONF_WEIGHTS.size))
        val direction = CodecAutoVec.tanhStable(directionLogit)
        var confidence = CodecAutoVec.sigmoid(confLogit)

        if (features.size >= 5) {
            val tail = DoubleArray(5)
            val weights = doubleArrayOf(0.10, 0.15, 0.20, 0.25, 0.30)
            var k = 0
            val start = features.size - 5
            while (k < 5) {
                tail[k] = features[start + k]
                k += 1
            }
            val weightedRet = CodecAutoVec.weightedAverage(tail, weights)
            val momentumBias = CodecAutoVec.sign(weightedRet)
            if (momentumBias != 0.0 && CodecAutoVec.sign(direction) != momentumBias) {
                confidence *= 0.7
            }
        }

        return signal(input, 
            confidence = confidence,
            direction = direction,
            instruments = mapOf(
                "attn_dir_logit" to directionLogit,
                "attn_conf_logit" to confLogit,
            ),
        )
    }

    private fun evalZscoreStatArb(input: CodecInput): CodecSignal {
        val returns = featureWindow(input.features)
        if (returns.size < 20) {
            return gateSignal(
                input = input,
                gateReasonCode = GateReason.INSUFFICIENT_FEATURE_HISTORY,
                confidence = 0.1,
                extras = mapOf(
                    "required_feature_count" to 20.0,
                    "observed_feature_count" to returns.size.toDouble(),
                ),
            )
        }
        val prices = CodecAutoVec.resolveOhlcv(input).closes
        if (prices.size < 20) {
            return gateSignal(
                input = input,
                gateReasonCode = GateReason.INSUFFICIENT_OHLCV_HISTORY,
                confidence = 0.1,
                extras = mapOf(
                    "required_close_count" to 20.0,
                    "observed_close_count" to prices.size.toDouble(),
                ),
            )
        }
        val price = CodecAutoVec.market(input, "price", prices[prices.lastIndex])

        val z10 = CodecAutoVec.rollingZScore(prices, 10)
        val z20 = CodecAutoVec.rollingZScore(prices, 20)
        val z60 = CodecAutoVec.rollingZScore(prices, min(60, prices.size))
        val compositeZ = z10 * 0.5 + z20 * 0.3 + z60 * 0.2

        val channelSpread = emaChannelSpread(prices, 10, 30)
        val hlStart = CodecAutoVec.tailStart(prices.size, 60)
        val hlWindow = DoubleArray(prices.size - hlStart)
        var i = 0
        while (i < hlWindow.size) {
            hlWindow[i] = prices[hlStart + i]
            i += 1
        }
        val halfLife = halfLife(hlWindow)
        val reverting = halfLife < 40.0
        val pctRank = CodecAutoVec.percentileRank(prices, price)
        val totalZ = compositeZ * 0.6 + channelSpread * 0.4

        val (direction, confidence) = if (abs(totalZ) < 1.5) {
            0.0 to 0.15
        } else {
            val dir = -CodecAutoVec.sign(totalZ)
            val rawConviction = min(1.0, (abs(totalZ) - 1.5) / 2.0 + 0.3)
            val regimeFactor = if (reverting) 0.7 else 0.35
            val pctFactor = 1.0 + 0.3 * (abs(pctRank - 0.5) * 2.0)
            val conf = min(1.0, rawConviction * regimeFactor * pctFactor)
            dir to conf
        }

        return signal(input, 
            confidence = confidence,
            direction = direction,
            instruments = mapOf(
                "composite_z" to compositeZ,
                "channel_spread" to channelSpread,
                "half_life" to min(halfLife, 999.0),
                "pct_rank" to pctRank,
            ),
        )
    }

    private fun weightedAutocorr(values: DoubleArray, start: Int, endExclusive: Int, maxLag: Int): Double {
        val n = endExclusive - start
        if (n < maxLag + 2 || maxLag <= 0) return 0.0
        var weightSum = 0.0
        var weighted = 0.0
        var lag = 1
        while (lag <= maxLag) {
            val ac = autocorr(values, start, endExclusive, lag)
            val w = 1.0 / lag.toDouble()
            weighted += ac * w
            weightSum += w
            lag += 1
        }
        return CodecAutoVec.safeDiv(weighted, weightSum + 1e-8, 0.0)
    }

    private fun autocorr(values: DoubleArray, start: Int, endExclusive: Int, lag: Int): Double {
        val n = endExclusive - start
        if (n <= lag) return 0.0
        val mu = CodecAutoVec.mean(values, start, endExclusive)
        var denom = 0.0
        var num = 0.0
        var i = start
        while (i < endExclusive) {
            val d = values[i] - mu
            denom += d * d
            i += 1
        }
        if (denom < 1e-10) return 0.0
        i = start
        while (i + lag < endExclusive) {
            num += (values[i] - mu) * (values[i + lag] - mu)
            i += 1
        }
        return num / denom
    }

    private fun garmanKlassVol(): Double {
        if (lmCount < 2) return 0.0
        var hlSum = 0.0
        var ccSum = 0.0
        var ccCount = 0
        var i = 0
        while (i < lmCount) {
            val h = lmHighs[i]
            val l = lmLows[i]
            if (h > 0.0 && l > 0.0) {
                val lnHl = ln(h / (l + 1e-8))
                hlSum += lnHl * lnHl
            }
            if (i > 0 && lmCloses[i] > 0.0 && lmCloses[i - 1] > 0.0) {
                val lnCc = ln(lmCloses[i] / (lmCloses[i - 1] + 1e-8))
                ccSum += lnCc * lnCc
                ccCount += 1
            }
            i += 1
        }
        val n = max(1, min(lmCount, ccCount))
        val hlMean = hlSum / n.toDouble()
        val ccMean = ccSum / n.toDouble()
        val gk = 0.5 * hlMean - (2.0 * ln(2.0) - 1.0) * ccMean
        return sqrt(max(gk, 0.0))
    }

    private fun alphaMomentum(prices: DoubleArray): Pair<Double, Double> {
        if (prices.size < 26) return 0.0 to 0.0
        val emaFast = CodecAutoVec.ema(prices, 12)
        val emaSlow = CodecAutoVec.ema(prices, 26)
        val cross = CodecAutoVec.safeDiv(emaFast - emaSlow, emaSlow + 1e-8, 0.0)
        val signal = CodecAutoVec.tanhStable(cross * 50.0)
        val quality = min(1.0, abs(cross) * 100.0 + 0.2)
        return signal to quality
    }

    private fun alphaRsi(returns: DoubleArray): Pair<Double, Double> {
        val rsi = CodecAutoVec.rsi(returns, 14)
        return when {
            rsi < 30.0 -> 1.0 to ((30.0 - rsi) / 30.0)
            rsi > 70.0 -> -1.0 to ((rsi - 70.0) / 30.0)
            else -> 0.0 to 0.1
        }
    }

    private fun alphaBollinger(prices: DoubleArray): Pair<Double, Double> {
        if (prices.isEmpty()) return 0.0 to 0.0
        val start = CodecAutoVec.tailStart(prices.size, 20)
        val mu = CodecAutoVec.mean(prices, start, prices.size)
        val sd = CodecAutoVec.std(prices, start, prices.size) + 1e-8
        val bb = (prices[prices.lastIndex] - mu) / (2.0 * sd)
        if (abs(bb) < 0.7) return 0.0 to 0.1
        val signal = if (abs(bb) > 1.0) CodecAutoVec.sign(bb) else -CodecAutoVec.sign(bb)
        val quality = min(1.0, (abs(bb) - 0.7) * 0.8 + 0.2)
        return signal to quality
    }

    private fun alphaDonchian(prices: DoubleArray): Pair<Double, Double> {
        if (prices.size < 20) return 0.0 to 0.0
        val start = prices.size - 20
        var chMin = prices[start]
        var chMax = prices[start]
        var i = start + 1
        while (i < prices.size) {
            val p = prices[i]
            if (p < chMin) chMin = p
            if (p > chMax) chMax = p
            i += 1
        }
        val channel = (chMax - chMin) + 1e-8
        val pos = (prices[prices.lastIndex] - chMin) / channel
        val signal = 2.0 * pos - 1.0
        val quality = min(1.0, abs(pos - 0.5) * 2.5 * 0.8 + 0.1)
        return signal to quality
    }

    private fun alphaVolumeMomentum(returns: DoubleArray, price: Double, volume: Double): Pair<Double, Double> {
        val ret1 = if (returns.isNotEmpty()) returns[returns.lastIndex] else 0.0
        val volStart = CodecAutoVec.tailStart(returns.size, 20)
        val histVol = max(CodecAutoVec.std(returns, volStart, returns.size), 0.01)
        val volNorm = volume / (histVol * 1e6 + 1e-8)
        val signal = CodecAutoVec.sign(ret1) * min(1.0, sqrt(volNorm) * 0.5)
        val quality = min(1.0, abs(ret1) / (histVol + 1e-8) * 0.3 + 0.15 + abs(price) * 0.0)
        return signal to quality
    }

    private fun kalmanUpdate(price: Double): Pair<Double, Double> {
        val q = 0.01
        val r = 0.1

        val xPredPrice = kalmanXPrice + kalmanXVelocity
        val xPredVelocity = kalmanXVelocity

        val pPred00 = p00 + p01 + p10 + p11 + q
        val pPred01 = p01 + p11
        val pPred10 = p10 + p11
        val pPred11 = p11 + q

        val innovation = price - xPredPrice
        val s = pPred00 + r
        val k0 = CodecAutoVec.safeDiv(pPred00, s + 1e-8, 0.0)
        val k1 = CodecAutoVec.safeDiv(pPred10, s + 1e-8, 0.0)

        kalmanXPrice = xPredPrice + k0 * innovation
        kalmanXVelocity = xPredVelocity + k1 * innovation

        p00 = (1.0 - k0) * pPred00
        p01 = (1.0 - k0) * pPred01
        p10 = pPred10 - k1 * pPred00
        p11 = pPred11 - k1 * pPred01

        return kalmanXPrice to kalmanXVelocity
    }

    private fun estimateHurst(returns: DoubleArray): Double {
        if (returns.size < 20) return 0.5
        val maxLag = min(returns.size / 2, 49)
        if (maxLag < 3) return 0.5

        val lags = DoubleArray(maxLag - 1)
        val tau = DoubleArray(maxLag - 1)
        var idx = 0
        var lag = 2
        while (lag <= maxLag) {
            lags[idx] = lag.toDouble()
            val pairs = returns.size - lag
            if (pairs <= 1) {
                tau[idx] = 1e-8
            } else {
                val diffs = DoubleArray(pairs)
                var i = 0
                while (i < pairs) {
                    diffs[i] = returns[i + lag] - returns[i]
                    i += 1
                }
                tau[idx] = max(CodecAutoVec.std(diffs), 1e-8)
            }
            lag += 1
            idx += 1
        }

        val logLags = DoubleArray(lags.size)
        val logTau = DoubleArray(tau.size)
        var i = 0
        while (i < lags.size) {
            logLags[i] = ln(lags[i])
            logTau[i] = ln(tau[i])
            i += 1
        }
        val slope = linearRegressionSlope(logLags, logTau)
        return slope / 2.0
    }

    private fun linearRegressionSlope(x: DoubleArray, y: DoubleArray): Double {
        val n = min(x.size, y.size)
        if (n <= 1) return 0.0
        val mx = CodecAutoVec.mean(x, 0, n)
        val my = CodecAutoVec.mean(y, 0, n)
        var cov = 0.0
        var varX = 0.0
        var i = 0
        while (i < n) {
            val dx = x[i] - mx
            val dy = y[i] - my
            cov += dx * dy
            varX += dx * dx
            i += 1
        }
        return CodecAutoVec.safeDiv(cov, varX + 1e-8, 0.0)
    }

    private fun buildRandomForestFeatures(input: CodecInput): DoubleArray {
        val price = CodecAutoVec.market(input, "price", 1.0)
        val high = CodecAutoVec.market(input, "high", price)
        val low = CodecAutoVec.market(input, "low", price)
        val volume = CodecAutoVec.market(input, "volume", 1.0)
        val returns = featureWindow(input.features)
        val ohlcv = CodecAutoVec.resolveOhlcv(input)
        val prices = ohlcv.closes

        val p5 = if (prices.size >= 5) CodecAutoVec.mean(prices, prices.size - 5, prices.size) else price
        val p20 = if (prices.size >= 20) CodecAutoVec.mean(prices, prices.size - 20, prices.size) else price
        val mom5 = CodecAutoVec.safeDiv(price - p5, p5 + 1e-8, 0.0)
        val mom20 = CodecAutoVec.safeDiv(price - p20, p20 + 1e-8, 0.0)
        val rsi14 = CodecAutoVec.rsi(returns, 14)

        val emaSliceStart = CodecAutoVec.tailStart(prices.size, 20)
        val ema5 = CodecAutoVec.ema(prices, 5, emaSliceStart, prices.size)
        val ema20 = CodecAutoVec.ema(prices, 20, emaSliceStart, prices.size)
        val emaRatio = CodecAutoVec.safeDiv(ema5, ema20 + 1e-8, 1.0)

        val volStart = CodecAutoVec.tailStart(returns.size, 20)
        val vol20 = CodecAutoVec.std(returns, volStart, returns.size)
        val atrRatio = CodecAutoVec.safeDiv(high - low, price + 1e-8, 0.0)

        val mean20 = if (prices.size >= 20) CodecAutoVec.mean(prices, prices.size - 20, prices.size) else price
        val std20 = if (prices.size >= 20) CodecAutoVec.std(prices, prices.size - 20, prices.size) else 1.0
        val closeZ = CodecAutoVec.safeDiv(price - mean20, std20 + 1e-8, 0.0)
        val meanAbsRet = max(abs(CodecAutoVec.mean(returns, volStart, returns.size)), 1e-8)
        val volumeRatio = volume / (meanAbsRet * 1e6 + 1e-8)

        return doubleArrayOf(mom5, mom20, rsi14, emaRatio, vol20, atrRatio, closeZ, volumeRatio)
    }

    private fun buildXgboostFeatures(input: CodecInput): DoubleArray {
        val price = CodecAutoVec.market(input, "price", 1.0)
        val high = CodecAutoVec.market(input, "high", price)
        val low = CodecAutoVec.market(input, "low", price)
        val returns = featureWindow(input.features)
        val prices = CodecAutoVec.resolveOhlcv(input).closes

        fun momentum(window: Int): Double {
            if (prices.size < window) return 0.0
            val base = CodecAutoVec.mean(prices, prices.size - window, prices.size)
            return CodecAutoVec.safeDiv(price - base, base + 1e-8, 0.0)
        }

        val mom3 = momentum(3)
        val mom10 = momentum(10)
        val mom30 = momentum(30)

        val emaFast = CodecAutoVec.ema(prices, 12)
        val emaSlow = CodecAutoVec.ema(prices, 26)
        val macdLine = emaFast - emaSlow
        val signalLine = CodecAutoVec.ema(prices, 9)
        val macdHist = macdLine - signalLine

        val rsiNorm = CodecAutoVec.rsi(returns, 14) / 100.0
        val vol20 = CodecAutoVec.std(returns, CodecAutoVec.tailStart(returns.size, 20), returns.size)
        val longVol = CodecAutoVec.std(returns, CodecAutoVec.tailStart(returns.size, 50), returns.size).let { if (it == 0.0) vol20 else it }
        val volRatio = CodecAutoVec.safeDiv(vol20, longVol + 1e-8, 1.0)

        val mean20 = if (prices.size >= 20) CodecAutoVec.mean(prices, prices.size - 20, prices.size) else price
        val std20 = if (prices.size >= 20) CodecAutoVec.std(prices, prices.size - 20, prices.size) else 1.0
        val bbPos = CodecAutoVec.safeDiv(price - mean20, 2.0 * std20 + 1e-8, 0.0)

        val mean10 = if (prices.size >= 10) CodecAutoVec.mean(prices, prices.size - 10, prices.size) else price
        val std10 = if (prices.size >= 10) CodecAutoVec.std(prices, prices.size - 10, prices.size) else 1.0
        val z10 = CodecAutoVec.safeDiv(price - mean10, std10 + 1e-8, 0.0)
        val mean30 = if (prices.size >= 30) CodecAutoVec.mean(prices, prices.size - 30, prices.size) else price
        val std30 = if (prices.size >= 30) CodecAutoVec.std(prices, prices.size - 30, prices.size) else 1.0
        val z30 = CodecAutoVec.safeDiv(price - mean30, std30 + 1e-8, 0.0)
        val atrNorm = CodecAutoVec.safeDiv(high - low, price + 1e-8, 0.0)

        return doubleArrayOf(mom3, mom10, mom30, macdHist, rsiNorm, volRatio, bbPos, z10, z30, atrNorm)
    }

    private fun buildAttentionTokens(input: CodecInput): Array<DoubleArray> {
        val prices = CodecAutoVec.resolveOhlcv(input).closes
        val nPatches = 12
        val patchSize = 5
        if (prices.size < 3) return Array(nPatches) { DoubleArray(15) }

        val factors = intArrayOf(1, 3, 12)
        val byTimeframe = Array(factors.size) { Array(nPatches) { DoubleArray(5) } }
        var tfIdx = 0
        while (tfIdx < factors.size) {
            val factor = factors[tfIdx]
            val needed = nPatches * factor
            val start = CodecAutoVec.tailStart(prices.size, needed)
            val chunks = DoubleArray(nPatches)
            var i = 0
            while (i < nPatches) {
                val chunkStart = min(prices.size - 1, start + i * factor)
                val chunkEnd = min(prices.size, chunkStart + factor)
                chunks[i] = CodecAutoVec.mean(prices, chunkStart, chunkEnd)
                i += 1
            }
            i = 0
            while (i < nPatches) {
                val close = chunks[i]
                val open = if (i > 0) chunks[i - 1] else close
                val delta = abs(close - open)
                val high = max(close, open) + delta * 0.5
                val low = min(close, open) - delta * 0.5
                val ret = CodecAutoVec.safeDiv(close - open, open + 1e-8, 0.0)
                byTimeframe[tfIdx][i][0] = open
                byTimeframe[tfIdx][i][1] = high
                byTimeframe[tfIdx][i][2] = low
                byTimeframe[tfIdx][i][3] = close
                byTimeframe[tfIdx][i][4] = ret
                i += 1
            }
            tfIdx += 1
        }

        val tokens = Array(nPatches) { DoubleArray(15) }
        var i = 0
        while (i < nPatches) {
            var cursor = 0
            tfIdx = 0
            while (tfIdx < byTimeframe.size) {
                var j = 0
                while (j < 5) {
                    tokens[i][cursor] = byTimeframe[tfIdx][i][j]
                    cursor += 1
                    j += 1
                }
                tfIdx += 1
            }
            val mu = CodecAutoVec.mean(tokens[i])
            val sd = CodecAutoVec.std(tokens[i]) + 1e-5
            var j = 0
            while (j < tokens[i].size) {
                tokens[i][j] = (tokens[i][j] - mu) / sd
                j += 1
            }
            i += 1
        }
        return tokens
    }

    private fun emaChannelSpread(prices: DoubleArray, fast: Int, slow: Int): Double {
        if (prices.size < slow) return 0.0
        val emaFast = CodecAutoVec.ema(prices, fast)
        val emaSlow = CodecAutoVec.ema(prices, slow)
        val midline = (emaFast + emaSlow) * 0.5
        val channel = abs(emaFast - emaSlow) + 1e-8
        return CodecAutoVec.safeDiv(prices[prices.lastIndex] - midline, channel, 0.0)
    }

    private fun halfLife(prices: DoubleArray): Double {
        if (prices.size < 10) return Double.POSITIVE_INFINITY
        var cov = 0.0
        var varLag = 0.0
        var i = 1
        while (i < prices.size) {
            val delta = prices[i] - prices[i - 1]
            val yLag = prices[i - 1]
            cov += delta * yLag
            varLag += yLag * yLag
            i += 1
        }
        if (varLag < 1e-8) return Double.POSITIVE_INFINITY
        val beta = cov / varLag
        if (beta >= 0.0) return Double.POSITIVE_INFINITY
        return -ln(2.0) / (beta + 1e-10)
    }
}

private val RF_BASE_TREES = arrayOf(
    DecisionTree(0, doubleArrayOf(-0.02, 0.0, 0.02), doubleArrayOf(-1.0, -0.5, 0.5, 1.0)),
    DecisionTree(1, doubleArrayOf(-0.01, 0.0, 0.01), doubleArrayOf(-1.0, -0.3, 0.3, 1.0)),
    DecisionTree(2, doubleArrayOf(30.0, 45.0, 55.0, 70.0), doubleArrayOf(-1.0, -0.5, 0.0, 0.5, 1.0)),
    DecisionTree(3, doubleArrayOf(0.98, 1.0, 1.02), doubleArrayOf(-1.0, -0.3, 0.3, 1.0)),
    DecisionTree(4, doubleArrayOf(0.005, 0.015, 0.03), doubleArrayOf(0.2, -0.2, -0.5, -1.0)),
    DecisionTree(5, doubleArrayOf(0.5, 1.0, 2.0), doubleArrayOf(0.0, -0.3, -0.6, -1.0)),
    DecisionTree(6, doubleArrayOf(-2.0, -0.5, 0.5, 2.0), doubleArrayOf(-1.0, -0.3, 0.3, 1.0)),
    DecisionTree(7, doubleArrayOf(0.7, 1.0, 1.5), doubleArrayOf(-0.3, 0.0, 0.3, 1.0)),
)

private val XGB_STUMPS = arrayOf(
    WeakLearner(0, 0.0, -0.30, 0.30),
    WeakLearner(1, 0.0, -0.30, 0.30),
    WeakLearner(2, 0.0, -0.24, 0.24),
    WeakLearner(3, 0.0, -0.27, 0.27),
    WeakLearner(4, 0.3, 0.15, 0.0),
    WeakLearner(4, 0.7, 0.0, -0.15),
    WeakLearner(6, -1.0, 0.24, 0.0),
    WeakLearner(6, 1.0, 0.0, -0.24),
    WeakLearner(5, 0.5, 0.09, -0.09),
    WeakLearner(7, -1.5, 0.18, 0.0),
    WeakLearner(7, 1.5, 0.0, -0.18),
    WeakLearner(8, -2.0, 0.15, 0.0),
    WeakLearner(8, 2.0, 0.0, -0.15),
    WeakLearner(9, 0.01, 0.06, -0.06),
    WeakLearner(0, -0.01, -0.12, 0.0),
    WeakLearner(0, 0.01, 0.0, 0.12),
    WeakLearner(1, -0.005, -0.09, 0.0),
    WeakLearner(1, 0.005, 0.0, 0.09),
    WeakLearner(2, -0.002, -0.06, 0.0),
    WeakLearner(2, 0.002, 0.0, 0.06),
    WeakLearner(3, -0.001, -0.075, 0.0),
    WeakLearner(3, 0.001, 0.0, 0.075),
    WeakLearner(4, 0.45, 0.045, 0.0),
    WeakLearner(4, 0.55, 0.0, -0.045),
    WeakLearner(6, -0.5, 0.06, 0.0),
    WeakLearner(6, 0.5, 0.0, -0.06),
    WeakLearner(7, -0.5, 0.045, 0.0),
    WeakLearner(7, 0.5, 0.0, -0.045),
    WeakLearner(5, 1.5, 0.0, -0.03),
    WeakLearner(9, 0.005, 0.03, -0.03),
)

private val ATTN_DIR_WEIGHTS = doubleArrayOf(
    -0.32, 0.28, -0.21, 0.35, 0.18,
    -0.15, 0.11, -0.09, 0.24, 0.13,
    -0.10, 0.09, -0.07, 0.18, 0.15,
)

private val ATTN_CONF_WEIGHTS = doubleArrayOf(
    0.22, 0.20, 0.17, 0.24, 0.19,
    0.16, 0.15, 0.13, 0.21, 0.18,
    0.14, 0.12, 0.11, 0.19, 0.17,
)
