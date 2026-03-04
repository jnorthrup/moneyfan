package borg.moneyfan.hrm.ane

interface HrmCoder16x16 {
    fun encodeScalar(input: Double): DoubleArray
}

class HrmCpu16x16Coder : HrmCoder16x16 {
    override fun encodeScalar(input: Double): DoubleArray {
        val out = DoubleArray(16 * 16)
        var i = 0
        while (i < out.size) {
            out[i] = input
            i += 1
        }
        return out
    }
}
