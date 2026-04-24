package vec.ops

import vec.assertClose
import vec.numericalGradient
import vec.tensorGaussian
import kotlin.test.Test
import kotlin.test.assertTrue

class GELUTest {

    @Test
    fun geluNearZeroIsSmall() {
        // GELU(0) = 0
        val x = vec.Tensor(intArrayOf(1, 1), floatArrayOf(0.0f))
        val y = gelu(x)
        assertTrue(kotlin.math.abs(y.data[0]) < 1e-6f, "GELU(0) ≈ 0 이어야 함: ${y.data[0]}")
    }

    @Test
    fun geluBackwardMatchesNumerical() {
        val x = tensorGaussian(intArrayOf(3, 4), std = 1.0f)
        // loss = sum(gelu(x)) → dL/dy = 1 everywhere
        val ones = FloatArray(x.numel) { 1.0f }

        val numericDX = numericalGradient(x) { xx -> gelu(xx).data.sum() }
        val analyticDX = geluBackward(x, ones)

        assertClose(analyticDX, numericDX, message = "gelu backward")
    }
}
