package borg.moneyfan.hrm.iomux

import kotlin.test.Test
import kotlin.test.assertEquals

class HrmVectorLoopsTest {
    @Test
    fun hadamard4_multiplies_all_inputs_without_shape_changes() {
        val a = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
        val b = doubleArrayOf(2.0, 3.0, 4.0, 5.0)
        val c = doubleArrayOf(3.0, 4.0, 5.0, 6.0)

        val out = hadamard4(a, b, c)

        assertEquals(6.0, out[0])
        assertEquals(24.0, out[1])
        assertEquals(60.0, out[2])
        assertEquals(120.0, out[3])
    }

    @Test
    fun weightedBlend4_matches_expected_linear_mix() {
        val v = doubleArrayOf(2.0, 4.0, 8.0, 16.0)
        val score = weightedBlend4(v, 0.5, 0.25, 0.125, 0.125)
        assertEquals(5.0, score)
    }

    @Test
    fun fillScalar_writes_every_slot() {
        val out = DoubleArray(16)
        fillScalar(out, 42.0)

        var i = 0
        while (i < out.size) {
            assertEquals(42.0, out[i])
            i += 1
        }
    }
}
