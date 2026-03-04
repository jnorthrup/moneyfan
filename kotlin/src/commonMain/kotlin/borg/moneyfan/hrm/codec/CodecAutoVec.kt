package borg.moneyfan.hrm.codec

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.math.tanh

internal data class OhlcvSeries(
    val closes: DoubleArray,
    val highs: DoubleArray,
    val lows: DoubleArray,
    val volumes: DoubleArray,
) {
    val size: Int get() = closes.size
}

internal object CodecAutoVec {
    fun clip01(v: Double): Double = when {
        v < 0.0 -> 0.0
        v > 1.0 -> 1.0
        else -> v
    }

    fun clip11(v: Double): Double = when {
        v < -1.0 -> -1.0
        v > 1.0 -> 1.0
        else -> v
    }

    fun sigmoid(v: Double): Double = 1.0 / (1.0 + exp(-v))

    fun sign(v: Double): Double = when {
        v > 0.0 -> 1.0
        v < 0.0 -> -1.0
        else -> 0.0
    }

    fun safeDiv(num: Double, den: Double, fallback: Double = 0.0): Double {
        if (abs(den) < 1e-12) return fallback
        return num / den
    }

    fun market(input: CodecInput, key: String, fallback: Double): Double =
        input.market[key] ?: fallback

    fun tailStart(size: Int, window: Int): Int = if (size <= window) 0 else size - window

    fun sum(values: DoubleArray, start: Int = 0, endExclusive: Int = values.size): Double {
        var s = 0.0
        var i = max(0, start)
        val end = min(values.size, endExclusive)
        while (i < end) {
            s += values[i]
            i += 1
        }
        return s
    }

    fun mean(values: DoubleArray, start: Int = 0, endExclusive: Int = values.size): Double {
        val begin = max(0, start)
        val end = min(values.size, endExclusive)
        if (end <= begin) return 0.0
        return sum(values, begin, end) / (end - begin).toDouble()
    }

    fun std(values: DoubleArray, start: Int = 0, endExclusive: Int = values.size): Double {
        val begin = max(0, start)
        val end = min(values.size, endExclusive)
        val n = end - begin
        if (n <= 1) return 0.0
        val mu = mean(values, begin, end)
        var acc = 0.0
        var i = begin
        while (i < end) {
            val d = values[i] - mu
            acc += d * d
            i += 1
        }
        return sqrt(acc / n.toDouble())
    }

    fun dot(a: DoubleArray, b: DoubleArray, n: Int = min(a.size, b.size)): Double {
        var s = 0.0
        var i = 0
        val end = min(n, min(a.size, b.size))
        while (i < end) {
            s += a[i] * b[i]
            i += 1
        }
        return s
    }

    fun covarianceLag1(values: DoubleArray, start: Int = 0, endExclusive: Int = values.size): Double {
        val begin = max(0, start)
        val end = min(values.size, endExclusive)
        val n = end - begin
        if (n < 4) return 0.0
        val xStart = begin + 1
        val xEnd = end
        val yStart = begin
        val yEnd = end - 1
        val xMean = mean(values, xStart, xEnd)
        val yMean = mean(values, yStart, yEnd)
        var cov = 0.0
        var i = 0
        val pairs = min(xEnd - xStart, yEnd - yStart)
        while (i < pairs) {
            cov += (values[xStart + i] - xMean) * (values[yStart + i] - yMean)
            i += 1
        }
        return cov / pairs.toDouble()
    }

    fun covariance(a: DoubleArray, b: DoubleArray, n: Int = min(a.size, b.size)): Double {
        if (n <= 1) return 0.0
        val end = min(n, min(a.size, b.size))
        val ma = mean(a, 0, end)
        val mb = mean(b, 0, end)
        var cov = 0.0
        var i = 0
        while (i < end) {
            cov += (a[i] - ma) * (b[i] - mb)
            i += 1
        }
        return cov / end.toDouble()
    }

    fun variance(values: DoubleArray, start: Int = 0, endExclusive: Int = values.size): Double {
        val st = std(values, start, endExclusive)
        return st * st
    }

    fun ema(values: DoubleArray, span: Int, start: Int = 0, endExclusive: Int = values.size): Double {
        val begin = max(0, start)
        val end = min(values.size, endExclusive)
        if (end <= begin) return 0.0
        val alpha = 2.0 / (span.toDouble() + 1.0)
        var v = values[begin]
        var i = begin + 1
        while (i < end) {
            v = alpha * values[i] + (1.0 - alpha) * v
            i += 1
        }
        return v
    }

    fun rsi(returns: DoubleArray, period: Int = 14): Double {
        if (returns.isEmpty()) return 50.0
        val start = tailStart(returns.size, period)
        var gain = 0.0
        var loss = 0.0
        var gainCount = 0
        var lossCount = 0
        var i = start
        while (i < returns.size) {
            val r = returns[i]
            if (r > 0.0) {
                gain += r
                gainCount += 1
            } else if (r < 0.0) {
                loss += -r
                lossCount += 1
            }
            i += 1
        }
        val avgGain = if (gainCount > 0) gain / gainCount.toDouble() else 0.0
        val avgLoss = if (lossCount > 0) loss / lossCount.toDouble() else 1e-8
        val rs = avgGain / (avgLoss + 1e-8)
        return 100.0 - 100.0 / (1.0 + rs)
    }

    fun rollingZScore(values: DoubleArray, window: Int): Double {
        if (values.isEmpty()) return 0.0
        val start = tailStart(values.size, window)
        val mu = mean(values, start, values.size)
        val sd = std(values, start, values.size)
        return safeDiv(values[values.lastIndex] - mu, sd + 1e-8, 0.0)
    }

    fun percentileRank(values: DoubleArray, value: Double): Double {
        if (values.isEmpty()) return 0.5
        var count = 0
        var i = 0
        while (i < values.size) {
            if (values[i] < value) count += 1
            i += 1
        }
        return count.toDouble() / values.size.toDouble()
    }

    fun linearSlope(values: DoubleArray, start: Int = 0, endExclusive: Int = values.size): Double {
        val begin = max(0, start)
        val end = min(values.size, endExclusive)
        val n = end - begin
        if (n <= 1) return 0.0

        val xMean = (n - 1).toDouble() * 0.5
        val yMean = mean(values, begin, end)
        var cov = 0.0
        var varX = 0.0
        var i = 0
        while (i < n) {
            val dx = i.toDouble() - xMean
            val dy = values[begin + i] - yMean
            cov += dx * dy
            varX += dx * dx
            i += 1
        }
        return safeDiv(cov, varX + 1e-8, 0.0)
    }

    fun weightedAverage(values: DoubleArray, weights: DoubleArray): Double {
        if (values.isEmpty() || weights.isEmpty()) return 0.0
        val n = min(values.size, weights.size)
        var num = 0.0
        var den = 0.0
        var i = 0
        while (i < n) {
            num += values[i] * weights[i]
            den += weights[i]
            i += 1
        }
        return safeDiv(num, den + 1e-8, 0.0)
    }

    fun softmax(scores: DoubleArray): DoubleArray {
        if (scores.isEmpty()) return DoubleArray(0)
        var maxScore = scores[0]
        var i = 1
        while (i < scores.size) {
            if (scores[i] > maxScore) maxScore = scores[i]
            i += 1
        }

        val out = DoubleArray(scores.size)
        var sum = 0.0
        i = 0
        while (i < scores.size) {
            val e = exp(scores[i] - maxScore)
            out[i] = e
            sum += e
            i += 1
        }
        val inv = if (sum > 0.0) 1.0 / sum else 0.0
        i = 0
        while (i < out.size) {
            out[i] *= inv
            i += 1
        }
        return out
    }

    fun resolveOhlcv(input: CodecInput, window: Int = 64): OhlcvSeries {
        val price = market(input, "price", 1.0)
        val high = market(input, "high", price)
        val low = market(input, "low", price)
        val volume = market(input, "volume", 1.0)

        val closes = input.closes
        if (closes != null && closes.size > 1) {
            val highs = input.highs ?: DoubleArray(closes.size) { high }
            val lows = input.lows ?: DoubleArray(closes.size) { low }
            val volumes = input.volumes ?: DoubleArray(closes.size) { volume }
            return OhlcvSeries(
                closes = closes.copyOf(),
                highs = if (highs.size == closes.size) highs.copyOf() else DoubleArray(closes.size) { high },
                lows = if (lows.size == closes.size) lows.copyOf() else DoubleArray(closes.size) { low },
                volumes = if (volumes.size == closes.size) volumes.copyOf() else DoubleArray(closes.size) { volume },
            )
        }

        val n = min(max(input.features.size, 1), window)
        val returns = input.features
        val recon = DoubleArray(n)
        var running = 0.0
        var idx = n - 1
        while (idx >= 0) {
            val r = returns[idx]
            running += r
            recon[idx] = price * exp(-running)
            idx -= 1
        }
        if (recon.isNotEmpty()) {
            recon[recon.lastIndex] = price
        }

        return OhlcvSeries(
            closes = recon,
            highs = DoubleArray(n) { high },
            lows = DoubleArray(n) { low },
            volumes = DoubleArray(n) { volume },
        )
    }

    fun barRange(high: Double, low: Double): Double = max(1e-8, high - low)

    fun log1pRatio(current: Double, previous: Double): Double {
        if (current <= 0.0 || previous <= 0.0) return 0.0
        val c = ln(current + 1.0)
        val p = ln(previous + 1.0)
        if (abs(p) < 1e-12) return 0.0
        return (c / p) - 1.0
    }

    fun tanhStable(v: Double): Double = tanh(v)
}
