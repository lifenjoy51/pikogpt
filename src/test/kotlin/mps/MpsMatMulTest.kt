package mps

import mps.ops.mpsMatmul
import mps.ops.mpsMatmulBackward
import org.junit.jupiter.api.Assumptions
import turbo.TurboTensor
import turbo.ops.turboMatmul
import turbo.ops.turboMatmulBackward
import kotlin.math.abs
import kotlin.math.max
import kotlin.random.Random
import kotlin.test.Test

/**
 * mps Metal MatMul vs turbo CPU SIMD 동등성. macOS + dylib 가용 시에만 실행, 그 외에는 skip.
 *
 * float32 lane/thread 누적 순서가 달라 동일 비트는 아니지만, |delta| / max(|c|) ≤ 1e-4 보장.
 * 모델 학습 영향(loss curve)을 검증하는 게 아니라 ops 단위 동등성만 확인 — 1초 이내 유지.
 */
class MpsMatMulTest {

    private fun ensure() {
        val ok = MpsAvailability.ensureChecked()
        if (!ok) {
            println("[mps-test] unavailable: ${MpsAvailability.reason}")
            println("[mps-test] os.name=${System.getProperty("os.name")} os.arch=${System.getProperty("os.arch")} java.library.path=${System.getProperty("java.library.path")}")
        }
        Assumptions.assumeTrue(ok, "mps unavailable: ${MpsAvailability.reason}")
    }

    private fun randTensor(rows: Int, cols: Int, seed: Long): TurboTensor {
        val rng = Random(seed)
        val data = FloatArray(rows * cols) { rng.nextFloat() * 2f - 1f }
        return TurboTensor(intArrayOf(rows, cols), data)
    }

    private fun assertClose(label: String, expected: FloatArray, actual: FloatArray, rtol: Float = 1e-4f) {
        require(expected.size == actual.size) { "$label size mismatch: ${expected.size} vs ${actual.size}" }
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
    fun forwardMatchesTurbo() {
        ensure()
        // 작은 shape(naive 경로) + tiled 경로(>=16) + boundary(비배수) 모두 커버.
        val cases = listOf(
            Triple(4, 16, 16),    // naive (M<16)
            Triple(4, 32, 64),    // naive (M<16)
            Triple(8, 32, 1024),  // naive (M<16)
            Triple(1, 7, 13),     // naive (전부 작음)
            Triple(13, 5, 11),    // naive (전부 작음)
            Triple(32, 16, 16),   // tiled — 정확히 1 tile
            Triple(64, 64, 64),   // tiled — 4x4 tile
            Triple(33, 17, 31),   // tiled boundary — 16 배수 아님
            Triple(64, 256, 256), // 10M 모델 QKV proj shape
        )
        for ((m, k, n) in cases) {
            val a = randTensor(m, k, seed = (m * 131 + k * 17 + n).toLong())
            val b = randTensor(k, n, seed = (m * 7 + k * 23 + n * 5).toLong())
            val expected = turboMatmul(a, b).data
            val actual = mpsMatmul(a, b).data
            assertClose("forward[$m,$k,$n]", expected, actual)
        }
    }

    @Test
    fun backwardMatchesTurbo() {
        ensure()
        val cases = listOf(
            Triple(4, 16, 16),
            Triple(4, 32, 64),
            Triple(8, 32, 1024),
            Triple(3, 5, 7),
            Triple(32, 16, 16),   // tiled
            Triple(64, 64, 64),   // tiled
            Triple(33, 17, 31),   // tiled boundary
        )
        for ((m, k, n) in cases) {
            val a1 = randTensor(m, k, seed = (m + k + n + 100).toLong())
            val b1 = randTensor(k, n, seed = (m + k + n + 200).toLong())
            val a2 = TurboTensor(a1.shape.copyOf(), a1.data.copyOf())
            val b2 = TurboTensor(b1.shape.copyOf(), b1.data.copyOf())
            val gy = FloatArray(m * n) { Random((m + k + n + 300).toLong() + it).nextFloat() * 2f - 1f }

            turboMatmulBackward(a1, b1, gy)
            mpsMatmulBackward(a2, b2, gy)

            assertClose("dA[$m,$k,$n]", a1.grad!!, a2.grad!!)
            assertClose("dB[$m,$k,$n]", b1.grad!!, b2.grad!!)
        }
    }
}
