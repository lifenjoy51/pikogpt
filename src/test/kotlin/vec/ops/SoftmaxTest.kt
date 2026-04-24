package vec.ops

import vec.assertClose
import vec.numericalGradient
import vec.tensorGaussian
import kotlin.test.Test
import kotlin.test.assertTrue

class SoftmaxTest {

    @Test
    fun softmaxForwardIsRowStochastic() {
        val x = tensorGaussian(intArrayOf(3, 5), std = 1.0f)
        val s = softmax(x)
        for (i in 0 until s.rows) {
            var sum = 0.0f
            for (j in 0 until s.cols) {
                val v = s[i, j]
                assertTrue(v in 0.0f..1.0f, "softmax 원소 [0,1] 범위 벗어남: $v")
                sum += v
            }
            assertTrue(kotlin.math.abs(sum - 1.0f) < 1e-5f, "행 합이 1이 아님: $sum")
        }
    }

    @Test
    fun softmaxBackwardMatchesNumerical() {
        val x = tensorGaussian(intArrayOf(2, 4), std = 0.5f)

        // 임의의 가중치로 loss 정의: loss = Σ_ij w_ij * s_ij
        val w = tensorGaussian(intArrayOf(2, 4), std = 1.0f)

        val numericDX = numericalGradient(x) { xx ->
            val s = softmax(xx)
            var l = 0.0f
            for (i in s.data.indices) l += s.data[i] * w.data[i]
            l
        }

        val s = softmax(x)
        val analyticDX = softmaxBackward(s, w.data)

        assertClose(analyticDX, numericDX, message = "softmax backward")
    }
}
