package turbo

import turbo.ops.turboRmsNormBackward
import turbo.ops.turboRmsNormForward
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * RMSNorm 단위 테스트. vec에 RMSNorm 없으므로 유한차분으로 backward 검증.
 *   y_ij = (x_ij / rms_i) * γ_j   where rms_i = √(mean(x²) + eps)
 */
class TurboRMSNormTest {

    @Test
    fun forwardMatchesManualFormula() {
        val rng = Random(7)
        val n = 3; val c = 6
        val data = FloatArray(n * c) { rng.nextFloat() - 0.5f }
        val gammaData = FloatArray(c) { 0.9f + 0.2f * rng.nextFloat() }

        val x = TurboTensor(intArrayOf(n, c), data.copyOf())
        val gamma = TurboTensor(intArrayOf(c), gammaData.copyOf())
        val (y, _) = turboRmsNormForward(x, gamma, eps = 1e-5f)

        for (i in 0 until n) {
            var sumSq = 0.0
            for (j in 0 until c) {
                val v = data[i * c + j]
                sumSq += (v * v).toDouble()
            }
            val rms = sqrt(sumSq / c + 1e-5)
            for (j in 0 until c) {
                val expected = (data[i * c + j].toDouble() / rms * gammaData[j]).toFloat()
                val diff = abs(expected - y.data[i * c + j])
                assertTrue(diff < 1e-5f, "row=$i col=$j expected=$expected got=${y.data[i * c + j]}")
            }
        }
    }

    @Test
    fun backwardMatchesFiniteDifference() {
        val rng = Random(9)
        val n = 2; val c = 5
        val data = FloatArray(n * c) { rng.nextFloat() - 0.5f }
        val gammaData = FloatArray(c) { 0.8f + 0.4f * rng.nextFloat() }
        val gyData = FloatArray(n * c) { rng.nextFloat() - 0.5f }
        val eps = 1e-5f

        // Analytic backward
        val xTensor = TurboTensor(intArrayOf(n, c), data.copyOf())
        val gammaTensor = TurboTensor(intArrayOf(c), gammaData.copyOf())
        val (_, cache) = turboRmsNormForward(xTensor, gammaTensor, eps)
        val dxAnalytic = turboRmsNormBackward(cache, gammaTensor, gyData.copyOf(), n, c)
        val dGammaAnalytic = gammaTensor.grad!!.copyOf()

        // Finite-difference dx
        val h = 1e-3f
        for (idx in data.indices) {
            val orig = data[idx]
            data[idx] = orig + h
            val (yPlus, _) = turboRmsNormForward(
                TurboTensor(intArrayOf(n, c), data.copyOf()),
                TurboTensor(intArrayOf(c), gammaData.copyOf()),
                eps,
            )
            data[idx] = orig - h
            val (yMinus, _) = turboRmsNormForward(
                TurboTensor(intArrayOf(n, c), data.copyOf()),
                TurboTensor(intArrayOf(c), gammaData.copyOf()),
                eps,
            )
            data[idx] = orig

            // L = Σ y_ij * gy_ij. dL/dx[idx] = Σ_ij gy_ij * (∂y_ij/∂x[idx])
            var lossPlus = 0.0
            var lossMinus = 0.0
            for (k in yPlus.data.indices) {
                lossPlus += (yPlus.data[k] * gyData[k]).toDouble()
                lossMinus += (yMinus.data[k] * gyData[k]).toDouble()
            }
            val numerical = ((lossPlus - lossMinus) / (2.0 * h)).toFloat()
            val analytic = dxAnalytic[idx]
            assertTrue(
                abs(numerical - analytic) < 5e-2f,
                "dx finite-diff mismatch at idx=$idx: numerical=$numerical, analytic=$analytic",
            )
        }

        // Finite-difference dgamma
        for (gIdx in gammaData.indices) {
            val orig = gammaData[gIdx]
            gammaData[gIdx] = orig + h
            val (yPlus, _) = turboRmsNormForward(
                TurboTensor(intArrayOf(n, c), data.copyOf()),
                TurboTensor(intArrayOf(c), gammaData.copyOf()),
                eps,
            )
            gammaData[gIdx] = orig - h
            val (yMinus, _) = turboRmsNormForward(
                TurboTensor(intArrayOf(n, c), data.copyOf()),
                TurboTensor(intArrayOf(c), gammaData.copyOf()),
                eps,
            )
            gammaData[gIdx] = orig

            var lossPlus = 0.0
            var lossMinus = 0.0
            for (k in yPlus.data.indices) {
                lossPlus += (yPlus.data[k] * gyData[k]).toDouble()
                lossMinus += (yMinus.data[k] * gyData[k]).toDouble()
            }
            val numerical = ((lossPlus - lossMinus) / (2.0 * h)).toFloat()
            val analytic = dGammaAnalytic[gIdx]
            assertTrue(
                abs(numerical - analytic) < 5e-2f,
                "dgamma mismatch at idx=$gIdx: numerical=$numerical, analytic=$analytic",
            )
        }
    }
}
