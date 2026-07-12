package borg.moneyfan.hrm.codec

import kotlin.test.Test
import kotlin.test.assertEquals

class CodecAutoVecTest {
    private val delta = 1e-12

    @Test
    fun percentileRank_emptyArray_returnsDefault() {
        val values = doubleArrayOf()
        val result = CodecAutoVec.percentileRank(values, 1.0)
        assertEquals(0.5, result, delta)
    }

    @Test
    fun percentileRank_singleElement_lessThan() {
        val values = doubleArrayOf(10.0)
        val result = CodecAutoVec.percentileRank(values, 5.0)
        assertEquals(0.0, result, delta)
    }

    @Test
    fun percentileRank_singleElement_greaterThan() {
        val values = doubleArrayOf(10.0)
        val result = CodecAutoVec.percentileRank(values, 15.0)
        assertEquals(1.0, result, delta)
    }

    @Test
    fun percentileRank_singleElement_equal() {
        val values = doubleArrayOf(10.0)
        val result = CodecAutoVec.percentileRank(values, 10.0)
        assertEquals(0.0, result, delta)
    }

    @Test
    fun percentileRank_multiElement_various() {
        val values = doubleArrayOf(10.0, 20.0, 30.0, 40.0, 50.0)
        assertEquals(0.0, CodecAutoVec.percentileRank(values, 5.0), delta)
        assertEquals(0.2, CodecAutoVec.percentileRank(values, 15.0), delta)
        assertEquals(0.4, CodecAutoVec.percentileRank(values, 25.0), delta)
        assertEquals(0.6, CodecAutoVec.percentileRank(values, 35.0), delta)
        assertEquals(0.8, CodecAutoVec.percentileRank(values, 45.0), delta)
        assertEquals(1.0, CodecAutoVec.percentileRank(values, 55.0), delta)
    }

    @Test
    fun percentileRank_duplicates() {
        val values = doubleArrayOf(10.0, 10.0, 20.0, 20.0, 30.0)
        // Values < 10.0: 0 -> 0/5 = 0.0
        assertEquals(0.0, CodecAutoVec.percentileRank(values, 10.0), delta)
        // Values < 15.0: 2 -> 2/5 = 0.4
        assertEquals(0.4, CodecAutoVec.percentileRank(values, 15.0), delta)
        // Values < 20.0: 2 -> 2/5 = 0.4
        assertEquals(0.4, CodecAutoVec.percentileRank(values, 20.0), delta)
        // Values < 25.0: 4 -> 4/5 = 0.8
        assertEquals(0.8, CodecAutoVec.percentileRank(values, 25.0), delta)
        // Values < 35.0: 5 -> 5/5 = 1.0
        assertEquals(1.0, CodecAutoVec.percentileRank(values, 35.0), delta)
    }
}
