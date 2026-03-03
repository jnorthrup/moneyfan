@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class, kotlin.experimental.ExperimentalNativeApi::class)
package borg.moneyfan

import kotlinx.cinterop.*

@CName("moneyfan_init_model")
fun initModel(path: CPointer<ByteVar>?): Int {
    val modelPath = path?.toKString() ?: return -1
    println("Initializing HRM model from $modelPath")
    return 0
}

@CName("moneyfan_predict")
fun predict(
    open: Double,
    high: Double,
    low: Double,
    close: Double,
    volume: Double
): Int {
    // Basic stochastic placeholder: return 1 (buy) if close > open, else -1 (sell)
    return if (close > open) 1 else -1
}

private fun CPointer<ByteVar>.toKString(): String? =
    this.readBytes().decodeToString().takeIf { it.isNotEmpty() }

private fun CPointer<ByteVar>.readBytes(): ByteArray {
    var length = 0
    while (this[length].toInt() != 0) {
        length++
    }
    val bytes = ByteArray(length)
    for (i in 0 until length) {
        bytes[i] = this[i]
    }
    return bytes
}
