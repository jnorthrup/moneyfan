package borg.moneyfan.hrm.ane

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class HrmAneSampleNet16x16CoderTest {
    @Test
    fun native_or_fallback_coder_produces_16x16_frame() {
        val coder = HrmNative16x16Coder()
        val out = coder.encodeScalar(2.25)
        assertEquals(16 * 16, out.size)
    }

    @Test
    fun sample_net_bridge_initializes_when_available() {
        if (!HrmAneSampleNet16x16Coder.available()) {
            return
        }

        val rc = HrmAneSampleNet16x16Coder.initialize()
        assertTrue(rc >= 0)

        val out = HrmAneSampleNet16x16Coder.encodeScalar(1.5)
        assertEquals(16 * 16, out.size)

        HrmAneSampleNet16x16Coder.close()
    }
}
