package turbo

import turbo.ops.turboMatmul
import turbo.ops.turboMatmulBackward
import vec.Tensor as VecTensor
import vec.ops.matmul as vecMatmul
import vec.ops.matmulBackward as vecMatmulBackward
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Phase 0 동등성: turbo MatMul forward/backward가 vec MatMul과 수치 일치 여부를 검증.
 * 같은 시드로 입력을 만들어 maxAbsDiff가 1e-4 이하여야 한다 (float32 누적 오차 마진).
 */
class TurboMatMulEquivalenceTest {

    @Test
    fun forwardMatchesVec() {
        val rng = Random(42)
        val m = 8; val k = 12; val n = 7
        val aData = FloatArray(m * k) { rng.nextFloat() - 0.5f }
        val bData = FloatArray(k * n) { rng.nextFloat() - 0.5f }

        val turboA = TurboTensor(intArrayOf(m, k), aData.copyOf())
        val turboB = TurboTensor(intArrayOf(k, n), bData.copyOf())
        val turboC = turboMatmul(turboA, turboB)

        val vecA = VecTensor(intArrayOf(m, k), aData.copyOf())
        val vecB = VecTensor(intArrayOf(k, n), bData.copyOf())
        val vecC = vecMatmul(vecA, vecB)

        val fwdDiff = maxAbsDiff(turboC.data, vecC.data)
        assertTrue(fwdDiff < 1e-4f, "MatMul forward 결과 mismatch: maxDiff=$fwdDiff")
    }

    @Test
    fun backwardMatchesVec() {
        val rng = Random(123)
        val m = 6; val k = 9; val n = 5
        val aData = FloatArray(m * k) { rng.nextFloat() - 0.5f }
        val bData = FloatArray(k * n) { rng.nextFloat() - 0.5f }
        val gyData = FloatArray(m * n) { rng.nextFloat() - 0.5f }

        val turboA = TurboTensor(intArrayOf(m, k), aData.copyOf())
        val turboB = TurboTensor(intArrayOf(k, n), bData.copyOf())
        turboMatmulBackward(turboA, turboB, gyData.copyOf())

        val vecA = VecTensor(intArrayOf(m, k), aData.copyOf())
        val vecB = VecTensor(intArrayOf(k, n), bData.copyOf())
        vecMatmulBackward(vecA, vecB, gyData.copyOf())

        assertTrue(maxAbsDiff(turboA.grad!!, vecA.grad!!) < 1e-4f, "dA mismatch")
        assertTrue(maxAbsDiff(turboB.grad!!, vecB.grad!!) < 1e-4f, "dB mismatch")
    }

    private fun maxAbsDiff(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size)
        var maxD = 0.0f
        for (i in a.indices) {
            val d = abs(a[i] - b[i])
            if (d > maxD) maxD = d
        }
        return maxD
    }
}
