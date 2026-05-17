package mps

import mps.ops.mpsMatmulFp16
import org.junit.jupiter.api.Assumptions
import turbo.TurboTensor
import turbo.ops.turboMatmul
import kotlin.math.abs
import kotlin.math.max
import kotlin.random.Random
import kotlin.test.Test

/**
 * F PoC — fp16 forward 정확도. fp16 mma는 fp32 대비 ~1e-3 누적 오차 → 별도 rtol 5e-3.
 * 학습 안정성(loss curve)은 사용자 수동 검증 영역이고 여기선 ops 단위 동등성만 확인.
 */
class MpsMatMulFp16Test {

    private fun ensure() {
        val ok = MpsAvailability.ensureChecked()
        Assumptions.assumeTrue(ok, "mps unavailable: ${MpsAvailability.reason}")
    }

    private fun randTensor(rows: Int, cols: Int, seed: Long): TurboTensor {
        val rng = Random(seed)
        // fp16 range overflow 방지 (max ±65504). [-1, 1) 범위로 제한.
        val data = FloatArray(rows * cols) { rng.nextFloat() * 2f - 1f }
        return TurboTensor(intArrayOf(rows, cols), data)
    }

    private fun assertClose(label: String, expected: FloatArray, actual: FloatArray, rtol: Float) {
        require(expected.size == actual.size) { "$label size mismatch" }
        var maxAbs = 0f
        var maxDelta = 0f
        for (i in expected.indices) {
            val a = abs(expected[i])
            if (a > maxAbs) maxAbs = a
            val d = abs(expected[i] - actual[i])
            if (d > maxDelta) maxDelta = d
        }
        val tol = rtol * max(1f, maxAbs)
        check(maxDelta <= tol) { "$label: maxDelta=$maxDelta > tol=$tol (rtol=$rtol, maxAbs=$maxAbs)" }
    }

    @Test
    fun forwardFp16WithinLooseTolerance() {
        ensure()
        // 10M 모델 실제 forward shape만. K가 클수록 fp16 누적 오차 ↑, rtol 5e-3로 흡수.
        val cases = listOf(
            Triple(64, 64, 64),
            Triple(64, 256, 256),
            Triple(64, 256, 1024),
            Triple(64, 1024, 256),
        )
        for ((m, k, n) in cases) {
            val a = randTensor(m, k, seed = (m * 131 + k * 17 + n).toLong())
            val b = randTensor(k, n, seed = (m * 7 + k * 23 + n * 5).toLong())
            val expected = turboMatmul(a, b).data
            val actual = mpsMatmulFp16(a, b).data
            assertClose("fp16[$m,$k,$n]", expected, actual, rtol = 5e-3f)
        }
    }
}
