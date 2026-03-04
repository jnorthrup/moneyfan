package borg.moneyfan.hrm.iomux

import borg.moneyfan.hrm.HrmSwimlaneSpec
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.tanh

data class HrmIoFrame(
    val symbol: String = "GLOBAL",
    val open: Double,
    val high: Double,
    val low: Double,
    val close: Double,
    val volume: Double,
    val epochMillis: Long = 0L,
)

enum class HrmAction(val code: Int) {
    SELL(-1),
    HOLD(0),
    BUY(1),
}

data class HrmLaneSignal(
    val laneId: Int,
    val glyphTag: String,
    val score: Double,
    val weightedScore: Double,
    val weight: Double,
)

data class HrmMuxDecision(
    val symbol: String,
    val action: HrmAction,
    val score: Double,
    val confidence: Double,
    val volatility: Double,
    val laneSignals: List<HrmLaneSignal>,
    val epochMillis: Long,
)

data class HrmIOMuxerConfig(
    val buyThreshold: Double = 0.14,
    val sellThreshold: Double = -0.14,
    val holdBand: Double = 0.04,
    val minVolatilityForAction: Double = 0.0015,
    val scoreAmplifier: Double = 22.0,
    val confidenceScale: Double = 2.4,
    val macdWeight: Double = 0.70,
    val momentumWeight: Double = 0.20,
    val rangeWeight: Double = 0.07,
    val volumeWeight: Double = 0.03,
)

class HrmTrikeShedIOMuxer(
    private var config: HrmIOMuxerConfig = HrmIOMuxerConfig(),
    initialSwimlanes: List<HrmSwimlaneSpec> = defaultSwimlanes(),
) {
    private data class LaneState(
        var emaFast: Double = Double.NaN,
        var emaSlow: Double = Double.NaN,
        var emaSignal: Double = 0.0,
        var previousClose: Double = Double.NaN,
        var previousVolume: Double = 0.0,
    )

    private var swimlanes: List<HrmSwimlaneSpec> = initialSwimlanes.sortedBy { it.laneId }
    private var laneGlyphs: Map<Int, DoubleArray> = buildLaneGlyphs(swimlanes)
    private var laneTags: Map<Int, String> = buildLaneTags(swimlanes)

    private val symbolLaneState: MutableMap<String, MutableMap<Int, LaneState>> = mutableMapOf()

    var lastDecision: HrmMuxDecision? = null
        private set

    val laneCount: Int
        get() = swimlanes.size

    fun setConfig(newConfig: HrmIOMuxerConfig) {
        config = newConfig
    }

    fun configureSwimlanes(specs: List<HrmSwimlaneSpec>) {
        require(specs.isNotEmpty()) { "swimlane set must not be empty" }
        swimlanes = specs.sortedBy { it.laneId }
        laneGlyphs = buildLaneGlyphs(swimlanes)
        laneTags = buildLaneTags(swimlanes)
        reset()
    }

    fun reset() {
        symbolLaneState.clear()
        lastDecision = null
    }

    fun ingest(frame: HrmIoFrame): HrmMuxDecision {
        val symbol = frame.symbol.ifBlank { "GLOBAL" }
        val states = symbolLaneState.getOrPut(symbol) { mutableMapOf() }
        val laneSignals = ArrayList<HrmLaneSignal>(swimlanes.size)

        val normalizedOpen = when {
            frame.open != 0.0 -> frame.open
            frame.close != 0.0 -> frame.close
            else -> 1.0
        }
        val volatility = abs(frame.high - frame.low) / abs(normalizedOpen)
        val momentum = frame.close.deltaRatio(frame.open)

        var weightedScoreSum = 0.0
        var weightDenominator = 0.0

        for (lane in swimlanes) {
            val state = states.getOrPut(lane.laneId) { LaneState() }
            bootstrapLaneStateIfNeeded(state, frame.close, frame.volume)

            val alphaFast = alphaForPeriod(lane.fast)
            val alphaSlow = alphaForPeriod(lane.slow)
            val alphaSig = alphaForPeriod(lane.sig)

            state.emaFast += alphaFast * (frame.close - state.emaFast)
            state.emaSlow += alphaSlow * (frame.close - state.emaSlow)

            val macd = state.emaFast - state.emaSlow
            state.emaSignal += alphaSig * (macd - state.emaSignal)
            val macdHistogram = macd - state.emaSignal

            val closeMomentum = frame.close.deltaRatio(state.previousClose)
            val volumeImpulse = volumeImpulse(frame.volume, state.previousVolume)

            val glyph = laneGlyphs[lane.laneId] ?: NEUTRAL_GLYPH
            val glyphGain = glyphGain(glyph)
            val riskGain = riskTierGain(lane.riskTier)

            val raw = (macdHistogram * config.macdWeight) +
                (closeMomentum * config.momentumWeight) +
                (momentum * config.rangeWeight) +
                (volumeImpulse * config.volumeWeight)

            val laneScore = tanh(raw * lane.sharp * glyphGain * riskGain * config.scoreAmplifier)
            val weighted = laneScore * lane.weight

            weightedScoreSum += weighted
            weightDenominator += abs(lane.weight)
            laneSignals += HrmLaneSignal(
                laneId = lane.laneId,
                glyphTag = laneTags[lane.laneId] ?: "lane_${lane.laneId}",
                score = laneScore,
                weightedScore = weighted,
                weight = lane.weight,
            )

            state.previousClose = frame.close
            state.previousVolume = max(frame.volume, 0.0)
        }

        val aggregate = if (weightDenominator > 0.0) weightedScoreSum / weightDenominator else 0.0
        val action = resolveAction(aggregate, volatility)
        val confidence = min(1.0, abs(aggregate) * config.confidenceScale + min(volatility, 0.25))

        val decision = HrmMuxDecision(
            symbol = symbol,
            action = action,
            score = aggregate,
            confidence = confidence,
            volatility = volatility,
            laneSignals = laneSignals.toList(),
            epochMillis = frame.epochMillis,
        )
        lastDecision = decision
        return decision
    }

    private fun resolveAction(score: Double, volatility: Double): HrmAction {
        if (abs(score) < config.holdBand) return HrmAction.HOLD
        if (volatility < config.minVolatilityForAction) return HrmAction.HOLD
        return when {
            score >= config.buyThreshold -> HrmAction.BUY
            score <= config.sellThreshold -> HrmAction.SELL
            else -> HrmAction.HOLD
        }
    }

    private fun bootstrapLaneStateIfNeeded(state: LaneState, close: Double, volume: Double) {
        if (state.emaFast.isNaN()) {
            state.emaFast = close
            state.emaSlow = close
            state.emaSignal = 0.0
        }
        if (state.previousClose.isNaN()) {
            state.previousClose = close
        }
        if (state.previousVolume <= 0.0) {
            state.previousVolume = max(volume, 0.0)
        }
    }

    companion object {
        private val NEUTRAL_GLYPH = doubleArrayOf(1.0, 1.0, 1.0, 1.0)

        fun defaultSwimlanes(): List<HrmSwimlaneSpec> = listOf(
            HrmSwimlaneSpec(laneId = 0, archetype = "grid", riskTier = "protective", weight = 1.00, fast = 8.0, slow = 34.0, sig = 5.0, sharp = 0.85),
            HrmSwimlaneSpec(laneId = 1, archetype = "volatile_breakout", riskTier = "normal", weight = 1.05, fast = 6.0, slow = 24.0, sig = 4.0, sharp = 1.35),
            HrmSwimlaneSpec(laneId = 2, archetype = "trend", riskTier = "normal", weight = 1.00, fast = 12.0, slow = 26.0, sig = 9.0, sharp = 1.00),
            HrmSwimlaneSpec(laneId = 3, archetype = "mean_reversion", riskTier = "caution", weight = 0.95, fast = 14.0, slow = 30.0, sig = 10.0, sharp = 0.92),
            HrmSwimlaneSpec(laneId = 4, archetype = "momentum", riskTier = "normal", weight = 1.00, fast = 10.0, slow = 22.0, sig = 7.0, sharp = 1.12),
            HrmSwimlaneSpec(laneId = 5, archetype = "liquidity_rotation", riskTier = "caution", weight = 0.92, fast = 16.0, slow = 36.0, sig = 10.0, sharp = 0.90),
        )
    }
}

private fun buildLaneGlyphs(swimlanes: List<HrmSwimlaneSpec>): Map<Int, DoubleArray> = swimlanes.associate { lane ->
    lane.laneId to laneToGlyph(lane)
}

private fun buildLaneTags(swimlanes: List<HrmSwimlaneSpec>): Map<Int, String> = swimlanes.associate { lane ->
    lane.laneId to "${lane.archetype.trim().lowercase()}⊗${lane.riskTier.trim().lowercase()}⊗q"
}

private fun laneToGlyph(spec: HrmSwimlaneSpec): DoubleArray {
    val archetype = spec.archetype.trim().lowercase()
    val risk = spec.riskTier.trim().lowercase()

    val archetypeBasis = when (archetype) {
        "grid" -> doubleArrayOf(0.90, 1.10, 1.05, 0.90)
        "volatile_breakout" -> doubleArrayOf(1.20, 0.85, 0.95, 1.25)
        "trend" -> doubleArrayOf(1.10, 1.00, 1.00, 1.05)
        "mean_reversion" -> doubleArrayOf(0.95, 1.10, 1.15, 0.90)
        "momentum" -> doubleArrayOf(1.15, 0.95, 0.95, 1.10)
        "liquidity_rotation" -> doubleArrayOf(1.00, 1.05, 0.95, 1.00)
        else -> doubleArrayOf(1.0, 1.0, 1.0, 1.0)
    }

    val riskBasis = when (risk) {
        "protective" -> doubleArrayOf(0.90, 1.15, 1.15, 0.85)
        "caution" -> doubleArrayOf(0.95, 1.05, 1.10, 0.92)
        else -> doubleArrayOf(1.0, 1.0, 1.0, 1.0)
    }

    val tempo = (spec.fast / max(spec.slow, 1e-9)).coerceIn(0.05, 2.5)
    val sigRatio = (spec.sig / max(spec.slow, 1e-9)).coerceIn(0.01, 2.5)
    val quant = doubleArrayOf(
        spec.weight.coerceIn(0.1, 4.0),
        (1.0 / tempo).coerceIn(0.4, 2.5),
        sigRatio.coerceIn(0.2, 2.5),
        spec.sharp.coerceIn(0.1, 4.0),
    )

    return hadamard4(archetypeBasis, riskBasis, quant)
}

private fun alphaForPeriod(period: Double): Double {
    val p = period.coerceAtLeast(2.0)
    return 2.0 / (p + 1.0)
}

private fun glyphGain(glyph: DoubleArray): Double {
    val blended = weightedBlend4(glyph, 0.35, 0.25, 0.20, 0.20)
    return blended.coerceIn(0.50, 1.80)
}

private fun riskTierGain(riskTier: String): Double = when (riskTier.trim().lowercase()) {
    "protective" -> 0.78
    "caution" -> 0.90
    else -> 1.0
}

private fun volumeImpulse(current: Double, previous: Double): Double {
    if (current <= 0.0 || previous <= 0.0) return 0.0
    val c = ln(current + 1.0)
    val p = ln(previous + 1.0)
    if (p == 0.0) return 0.0
    return (c / p) - 1.0
}

private fun Double.deltaRatio(base: Double): Double {
    if (base == 0.0) return 0.0
    return (this - base) / abs(base)
}
