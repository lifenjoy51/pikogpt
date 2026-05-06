package turbo

import turbo.ops.turboCrossEntropyBackward
import turbo.ops.turboCrossEntropyForward
import kotlin.math.abs
import kotlin.math.ln
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * z-loss 검증.
 *   - zLossWeight=0 → 기존 CE와 정확히 동일 (회귀)
 *   - zLossWeight>0 → forward는 lse² 가산, backward는 +2w·lse·softmax 추가
 */
class TurboZLossTest {

    @Test
    fun zeroWeightMatchesPlainCrossEntropy() {
        val rng = Random(11)
        val n = 4; val v = 7
        val data = FloatArray(n * v) { rng.nextFloat() * 4f - 2f }
        val targets = IntArray(n) { rng.nextInt(v) }

        val x = TurboTensor(intArrayOf(n, v), data.copyOf())
        val res0 = turboCrossEntropyForward(x, targets, zLossWeight = 0.0f)
        val resPlain = turboCrossEntropyForward(x, targets)

        assertTrue(abs(res0.loss - resPlain.loss) < 1e-7f, "zLoss=0 forward should equal plain CE")

        val grad0 = turboCrossEntropyBackward(x, targets, res0.softmax, zLossWeight = 0.0f, lsePerRow = res0.lsePerRow)
        val gradPlain = turboCrossEntropyBackward(x, targets, resPlain.softmax)

        for (i in grad0.data.indices) {
            assertTrue(abs(grad0.data[i] - gradPlain.data[i]) < 1e-7f, "grad mismatch at $i")
        }
    }

    @Test
    fun nonZeroWeightForwardAddsLseSquared() {
        val rng = Random(12)
        val n = 3; val v = 5
        val data = FloatArray(n * v) { rng.nextFloat() * 4f - 2f }
        val targets = IntArray(n) { rng.nextInt(v) }
        val w = 0.01f

        val x = TurboTensor(intArrayOf(n, v), data.copyOf())
        val resPlain = turboCrossEntropyForward(x, targets)
        val resZ = turboCrossEntropyForward(x, targets, zLossWeight = w)

        // lse_i = log(Σ exp(logits_ij))
        var sumLseSq = 0.0
        for (i in 0 until n) {
            var maxV = Float.NEGATIVE_INFINITY
            for (j in 0 until v) if (data[i * v + j] > maxV) maxV = data[i * v + j]
            var sumExp = 0.0
            for (j in 0 until v) sumExp += kotlin.math.exp((data[i * v + j] - maxV).toDouble())
            val lse = ln(sumExp).toFloat() + maxV
            sumLseSq += (w * lse * lse).toDouble()
        }
        val expectedDelta = (sumLseSq / n).toFloat()
        val actualDelta = resZ.loss - resPlain.loss
        assertTrue(
            abs(actualDelta - expectedDelta) < 1e-4f,
            "z-loss forward delta mismatch: expected=$expectedDelta, actual=$actualDelta",
        )
    }

    @Test
    fun nonZeroWeightBackwardMatchesFiniteDifference() {
        val rng = Random(13)
        val n = 3; val v = 4
        val data = FloatArray(n * v) { rng.nextFloat() * 2f - 1f }
        val targets = IntArray(n) { rng.nextInt(v) }
        val w = 0.005f

        val xTensor = TurboTensor(intArrayOf(n, v), data.copyOf())
        val res = turboCrossEntropyForward(xTensor, targets, zLossWeight = w)
        val gAnalytic = turboCrossEntropyBackward(
            xTensor, targets, res.softmax, upstreamGrad = 1.0f,
            zLossWeight = w, lsePerRow = res.lsePerRow,
        )

        val h = 1e-3f
        for (idx in data.indices) {
            val orig = data[idx]
            data[idx] = orig + h
            val rPlus = turboCrossEntropyForward(
                TurboTensor(intArrayOf(n, v), data.copyOf()), targets, zLossWeight = w,
            )
            data[idx] = orig - h
            val rMinus = turboCrossEntropyForward(
                TurboTensor(intArrayOf(n, v), data.copyOf()), targets, zLossWeight = w,
            )
            data[idx] = orig

            val numerical = ((rPlus.loss - rMinus.loss) / (2.0f * h))
            val analytic = gAnalytic.data[idx]
            assertTrue(
                abs(numerical - analytic) < 5e-2f,
                "z-loss grad mismatch at idx=$idx: numerical=$numerical, analytic=$analytic",
            )
        }
    }
}
