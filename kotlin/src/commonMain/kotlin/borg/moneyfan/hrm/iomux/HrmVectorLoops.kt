package borg.moneyfan.hrm.iomux

internal const val HRM_VEC4 = 4

internal fun hadamard4(a: DoubleArray, b: DoubleArray, c: DoubleArray): DoubleArray {
    val out = DoubleArray(HRM_VEC4)
    var i = 0
    while (i < HRM_VEC4) {
        out[i] = a[i] * b[i] * c[i]
        i += 1
    }
    return out
}

internal fun weightedBlend4(v: DoubleArray, w0: Double, w1: Double, w2: Double, w3: Double): Double {
    return (v[0] * w0) +
        (v[1] * w1) +
        (v[2] * w2) +
        (v[3] * w3)
}

internal fun fillScalar(out: DoubleArray, value: Double) {
    var i = 0
    while (i < out.size) {
        out[i] = value
        i += 1
    }
}
