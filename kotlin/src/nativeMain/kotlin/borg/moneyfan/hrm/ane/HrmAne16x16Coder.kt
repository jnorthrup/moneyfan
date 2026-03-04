@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package borg.moneyfan.hrm.ane

import borg.moneyfan.ane.mf_ane_sample_net_available
import borg.moneyfan.ane.mf_ane_sample_net_close
import borg.moneyfan.ane.mf_ane_sample_net_eval_1x1_to_16x16
import borg.moneyfan.ane.mf_ane_sample_net_init
import kotlinx.cinterop.*

object HrmAneSampleNet16x16Coder {
    const val outputWidth: Int = 16
    const val outputHeight: Int = 16
    const val outputSize: Int = outputWidth * outputHeight

    fun available(): Boolean = mf_ane_sample_net_available() != 0

    fun initialize(): Int = mf_ane_sample_net_init()

    fun encodeScalar(input: Double): DoubleArray = memScoped {
        val inBuf = allocArray<FloatVar>(1)
        val outBuf = allocArray<FloatVar>(outputSize)
        inBuf[0] = input.toFloat()

        val rc = mf_ane_sample_net_eval_1x1_to_16x16(inBuf, outBuf)
        if (rc < 0) {
            error("ANE sample net 1x1->16x16 evaluation failed with code $rc")
        }

        val out = DoubleArray(outputSize)
        var i = 0
        while (i < outputSize) {
            out[i] = outBuf[i].toDouble()
            i += 1
        }
        out
    }

    fun close() {
        mf_ane_sample_net_close()
    }
}

typealias HrmAne16x16Coder = HrmAneSampleNet16x16Coder

class HrmNative16x16Coder(
    private val cpuFallback: HrmCoder16x16 = HrmCpu16x16Coder(),
) : HrmCoder16x16 {
    override fun encodeScalar(input: Double): DoubleArray {
        return if (HrmAneSampleNet16x16Coder.available()) {
            HrmAneSampleNet16x16Coder.encodeScalar(input)
        } else {
            cpuFallback.encodeScalar(input)
        }
    }
}
