package borg.moneyfan.hrm.ane

import kotlin.test.Test
import kotlin.test.assertEquals

class HrmCpu16x16CoderTest {
    @Test
    fun scalar_input_expands_to_16x16_frame() {
        val coder = HrmCpu16x16Coder()
        val out = coder.encodeScalar(7.5)

        assertEquals(16 * 16, out.size)

        var i = 0
        while (i < out.size) {
            assertEquals(7.5, out[i])
            i += 1
        }
    }
}
