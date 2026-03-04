package borg.moneyfan.hrm.ane

import kotlin.math.abs

data class SampleNetFitResult(
    val epochs: Int,
    val initialLoss: Double,
    val finalLoss: Double,
    val converged: Boolean,
    val learnedScale: Double,
    val learnedBias: Double,
)

class SampleNet16x16(
    private val coder: HrmCoder16x16,
    initialScale: Double = 0.0,
    initialBias: Double = 0.0,
) {
    var scale: Double = initialScale
        private set

    var bias: Double = initialBias
        private set

    fun predict(input: Double): Double {
        val frame = coder.encodeScalar(input)
        val mean = frameMean(frame)
        return (scale * mean) + bias
    }

    fun loss(inputs: DoubleArray, targets: DoubleArray): Double {
        require(inputs.size == targets.size) { "inputs and targets must have same size" }
        require(inputs.isNotEmpty()) { "inputs must not be empty" }

        var loss = 0.0
        var i = 0
        while (i < inputs.size) {
            val err = predict(inputs[i]) - targets[i]
            loss += err * err
            i += 1
        }
        return loss / inputs.size.toDouble()
    }

    fun trainEpoch(inputs: DoubleArray, targets: DoubleArray, learningRate: Double): Double {
        require(learningRate > 0.0) { "learningRate must be > 0" }
        require(inputs.size == targets.size) { "inputs and targets must have same size" }
        require(inputs.isNotEmpty()) { "inputs must not be empty" }

        val count = inputs.size.toDouble()

        var gradScale = 0.0
        var gradBias = 0.0
        var loss = 0.0

        var i = 0
        while (i < inputs.size) {
            val frame = coder.encodeScalar(inputs[i])
            val mean = frameMean(frame)
            val pred = (scale * mean) + bias
            val err = pred - targets[i]

            loss += err * err
            gradScale += err * mean
            gradBias += err

            i += 1
        }

        val gradScaleScaled = (2.0 / count) * gradScale
        val gradBiasScaled = (2.0 / count) * gradBias

        scale -= learningRate * gradScaleScaled
        bias -= learningRate * gradBiasScaled

        return loss / count
    }

    fun fit(
        inputs: DoubleArray,
        targets: DoubleArray,
        learningRate: Double,
        maxEpochs: Int,
        targetLoss: Double,
        minRelativeImprovement: Double = 1e-7,
    ): SampleNetFitResult {
        require(maxEpochs > 0) { "maxEpochs must be > 0" }
        require(targetLoss >= 0.0) { "targetLoss must be >= 0" }

        var epoch = 0
        var previousLoss = loss(inputs, targets)
        val initialLoss = previousLoss

        while (epoch < maxEpochs && previousLoss > targetLoss) {
            val currentLoss = trainEpoch(inputs, targets, learningRate)
            val relative = if (abs(previousLoss) > 1e-12) {
                abs(previousLoss - currentLoss) / abs(previousLoss)
            } else {
                abs(previousLoss - currentLoss)
            }
            previousLoss = currentLoss
            epoch += 1

            if (relative < minRelativeImprovement && currentLoss <= targetLoss * 10.0) {
                break
            }
        }

        val finalLoss = previousLoss
        return SampleNetFitResult(
            epochs = epoch,
            initialLoss = initialLoss,
            finalLoss = finalLoss,
            converged = finalLoss <= targetLoss,
            learnedScale = scale,
            learnedBias = bias,
        )
    }

    private fun frameMean(frame: DoubleArray): Double {
        require(frame.isNotEmpty()) { "coder output frame must not be empty" }
        var sum = 0.0
        var i = 0
        while (i < frame.size) {
            sum += frame[i]
            i += 1
        }
        return sum / frame.size.toDouble()
    }
}
