package borg.moneyfan.hrm.codec

data class CodecInput(
    val symbol: String = "GLOBAL",
    val market: Map<String, Double> = emptyMap(),
    val features: DoubleArray,
    val closes: DoubleArray? = null,
    val highs: DoubleArray? = null,
    val lows: DoubleArray? = null,
    val volumes: DoubleArray? = null,
)

data class CodecSignal(
    val slotId: Int,
    val slotName: String,
    val confidence: Double,
    val direction: Double,
    val instruments: Map<String, Double>,
)

interface CodecModel {
    val slotId: Int
    val slotName: String
    fun evaluate(input: CodecInput): CodecSignal
    fun reset() {}
}
