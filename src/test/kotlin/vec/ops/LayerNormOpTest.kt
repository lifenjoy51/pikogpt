package vec.ops

import vec.Tensor
import vec.assertClose
import vec.numericalGradient
import vec.tensorGaussian
import vec.tensorOnes
import vec.tensorZeros
import kotlin.test.Test
import kotlin.test.assertTrue

class LayerNormOpTest {

    @Test
    fun layerNormProducesZeroMeanUnitVariance() {
        val x = tensorGaussian(intArrayOf(3, 8), std = 1.5f)
        val gamma = tensorOnes(intArrayOf(8))
        val beta = tensorZeros(intArrayOf(8))

        val (y, _) = layerNormForward(x, gamma, beta)

        for (i in 0 until y.rows) {
            var mean = 0.0f
            for (j in 0 until y.cols) mean += y[i, j]
            mean /= y.cols
            var variance = 0.0f
            for (j in 0 until y.cols) {
                val d = y[i, j] - mean
                variance += d * d
            }
            variance /= y.cols
            assertTrue(kotlin.math.abs(mean) < 1e-4f, "행 평균 ≈ 0 실패: $mean")
            assertTrue(kotlin.math.abs(variance - 1.0f) < 1e-2f, "행 분산 ≈ 1 실패: $variance")
        }
    }

    @Test
    fun layerNormBackwardMatchesNumerical() {
        val x = tensorGaussian(intArrayOf(2, 5), std = 0.7f)
        val gamma = tensorGaussian(intArrayOf(5), std = 0.3f)
        val beta = tensorGaussian(intArrayOf(5), std = 0.3f)
        val w = tensorGaussian(intArrayOf(2, 5), std = 1.0f)

        // 수치 기울기: loss = Σ w * y (y = LN(x))
        val numericDX = numericalGradient(x) { xx ->
            val (yy, _) = layerNormForward(xx, gamma, beta)
            var l = 0.0f
            for (i in yy.data.indices) l += yy.data[i] * w.data[i]
            l
        }
        val numericDGamma = numericalGradient(gamma) { gg ->
            val (yy, _) = layerNormForward(x, gg, beta)
            var l = 0.0f
            for (i in yy.data.indices) l += yy.data[i] * w.data[i]
            l
        }
        val numericDBeta = numericalGradient(beta) { bb ->
            val (yy, _) = layerNormForward(x, gamma, bb)
            var l = 0.0f
            for (i in yy.data.indices) l += yy.data[i] * w.data[i]
            l
        }

        // 분석 backward
        val (_, cache) = layerNormForward(x, gamma, beta)
        val analyticDX = layerNormBackward(cache, gamma, beta, w.data, x.rows, x.cols)

        assertClose(analyticDX, numericDX, message = "dx")
        assertClose(gamma.grad!!, numericDGamma, message = "dγ")
        assertClose(beta.grad!!, numericDBeta, message = "dβ")
    }
}
