package borg.moneyfan.hrm.ane

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

class SampleNet16x16ConvergenceTest {
    @Test
    fun sample_net_converges_for_linear_target() {
        val coder = HrmCpu16x16Coder()
        val net = SampleNet16x16(
            coder = coder,
            initialScale = -0.8,
            initialBias = 0.9,
        )

        val samples = 192
        val inputs = DoubleArray(samples)
        val targets = DoubleArray(samples)

        val targetScale = 1.75
        val targetBias = -0.33

        var i = 0
        while (i < samples) {
            val x = -1.0 + (2.0 * i.toDouble() / (samples - 1).toDouble())
            inputs[i] = x
            targets[i] = (targetScale * x) + targetBias
            i += 1
        }

        val fit = net.fit(
            inputs = inputs,
            targets = targets,
            learningRate = 0.20,
            maxEpochs = 600,
            targetLoss = 1e-8,
        )

        assertTrue(fit.finalLoss < fit.initialLoss * 1e-5)
        assertTrue(fit.finalLoss < 1e-6)
        assertTrue(fit.converged)
        assertTrue(abs(fit.learnedScale - targetScale) < 1e-2)
        assertTrue(abs(fit.learnedBias - targetBias) < 1e-2)
    }
}
