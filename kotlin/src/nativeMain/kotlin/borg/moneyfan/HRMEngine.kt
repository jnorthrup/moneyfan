@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class, kotlin.experimental.ExperimentalNativeApi::class)
package borg.moneyfan

import borg.moneyfan.hrm.HrmSwimlaneDsl
import borg.moneyfan.hrm.ane.HrmAne16x16Coder
import borg.moneyfan.hrm.iomux.HrmAction
import borg.moneyfan.hrm.iomux.HrmIoFrame
import borg.moneyfan.hrm.iomux.HrmMuxDecision
import borg.moneyfan.hrm.iomux.HrmTrikeShedIOMuxer
import kotlinx.cinterop.*
import kotlin.native.concurrent.ThreadLocal

@ThreadLocal
private object HrmEngineState {
    val muxer = HrmTrikeShedIOMuxer()
    var modelPath: String = ""
    var lastDecision: HrmMuxDecision? = null
}

@CName("moneyfan_init_model")
fun initModel(path: CPointer<ByteVar>?): Int {
    val modelPath = path?.toKString()?.trim().orEmpty()
    if (modelPath.isBlank()) return -1

    HrmEngineState.modelPath = modelPath
    if (modelPath.endsWith(".dsl")) {
        val configured = runCatching {
            val specs = HrmSwimlaneDsl.load(modelPath)
            HrmSwimlaneDsl.requireArchetypes(specs, setOf("grid", "volatile_breakout"))
            HrmEngineState.muxer.configureSwimlanes(specs)
        }
        if (configured.isFailure) {
            println("Failed to configure swimlane dsl from $modelPath: ${configured.exceptionOrNull()?.message}")
            return -2
        }
    }
    println("Initialized HRM model context from $modelPath")
    return 0
}

@CName("moneyfan_predict")
fun predict(
    open: Double,
    high: Double,
    low: Double,
    close: Double,
    volume: Double,
): Int = ingestAndCache(
    symbol = "GLOBAL",
    open = open,
    high = high,
    low = low,
    close = close,
    volume = volume,
    epochMillis = 0L,
).action.code

@CName("moneyfan_iomux_ingest")
fun iomuxIngest(
    symbol: CPointer<ByteVar>?,
    open: Double,
    high: Double,
    low: Double,
    close: Double,
    volume: Double,
    epochMillis: Long,
): Int {
    val sym = symbol?.toKString()?.trim().orEmpty().ifEmpty { "GLOBAL" }
    return ingestAndCache(
        symbol = sym,
        open = open,
        high = high,
        low = low,
        close = close,
        volume = volume,
        epochMillis = epochMillis,
    ).action.code
}

@CName("moneyfan_iomux_configure_swimlanes_dsl")
fun iomuxConfigureSwimlanesDsl(path: CPointer<ByteVar>?): Int {
    val manifestPath = path?.toKString()?.trim().orEmpty()
    if (manifestPath.isBlank()) return -1
    val configured = runCatching {
        val specs = HrmSwimlaneDsl.load(manifestPath)
        HrmEngineState.muxer.configureSwimlanes(specs)
    }
    return if (configured.isSuccess) 0 else -2
}

@CName("moneyfan_iomux_use_default_swimlanes")
fun iomuxUseDefaultSwimlanes(): Int {
    HrmEngineState.muxer.configureSwimlanes(HrmTrikeShedIOMuxer.defaultSwimlanes())
    HrmEngineState.lastDecision = null
    return 0
}

@CName("moneyfan_iomux_reset")
fun iomuxReset(): Int {
    HrmEngineState.muxer.reset()
    HrmEngineState.lastDecision = null
    return 0
}

@CName("moneyfan_iomux_last_score")
fun iomuxLastScore(): Double = HrmEngineState.lastDecision?.score ?: 0.0

@CName("moneyfan_iomux_last_confidence")
fun iomuxLastConfidence(): Double = HrmEngineState.lastDecision?.confidence ?: 0.0

@CName("moneyfan_iomux_last_volatility")
fun iomuxLastVolatility(): Double = HrmEngineState.lastDecision?.volatility ?: 0.0

@CName("moneyfan_iomux_last_lane_count")
fun iomuxLastLaneCount(): Int = HrmEngineState.lastDecision?.laneSignals?.size ?: HrmEngineState.muxer.laneCount

@CName("moneyfan_iomux_last_action")
fun iomuxLastAction(): Int = (HrmEngineState.lastDecision?.action ?: HrmAction.HOLD).code

@CName("moneyfan_hrm_ane_available")
fun hrmAneAvailable(): Int = if (HrmAne16x16Coder.available()) 1 else 0

@CName("moneyfan_hrm_ane_encode_scalar_16x16")
fun hrmAneEncodeScalar16x16(input: Double, output: CPointer<DoubleVar>?): Int {
    if (output == null) return -1

    return runCatching {
        val encoded = HrmAne16x16Coder.encodeScalar(input)
        var i = 0
        while (i < encoded.size) {
            output[i] = encoded[i]
            i += 1
        }
        encoded.size
    }.getOrElse { -2 }
}

@CName("moneyfan_sample_net_available")
fun sampleNetAvailable(): Int = hrmAneAvailable()

@CName("moneyfan_sample_net_encode_scalar_16x16")
fun sampleNetEncodeScalar16x16(input: Double, output: CPointer<DoubleVar>?): Int {
    return hrmAneEncodeScalar16x16(input, output)
}

private fun ingestAndCache(
    symbol: String,
    open: Double,
    high: Double,
    low: Double,
    close: Double,
    volume: Double,
    epochMillis: Long,
): HrmMuxDecision {
    val decision = HrmEngineState.muxer.ingest(
        HrmIoFrame(
            symbol = symbol,
            open = open,
            high = high,
            low = low,
            close = close,
            volume = volume,
            epochMillis = epochMillis,
        )
    )
    HrmEngineState.lastDecision = decision
    return decision
}
