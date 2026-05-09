package turbo.bench

import turbo.TurboSimdMath
import turbo.TurboTensor
import turbo.ops.turboMatmul
import kotlin.random.Random
import kotlin.system.measureNanoTime

/**
 * Turbo MatMul 마이크로벤치마크 — shape별 절대 처리 시간 측정.
 *
 * vec 백엔드 폐기 후 turbo 단독 측정. 회귀 추적용 (이전 측정과 절대 시간 비교).
 */
fun main() {
    println("=== Turbo MatMul micro-benchmark ===")
    println("SIMD lane count: ${TurboSimdMath.laneCount} (FloatVector.SPECIES_PREFERRED)")
    println("CPU cores: ${Runtime.getRuntime().availableProcessors()}")
    println()

    benchMatmul()
}

private fun benchMatmul() {
    println("--- MatMul forward ---")
    println("%-22s %12s".format("shape", "turbo (us)"))
    val warmupIters = 20
    val measureIters = 100

    val cases = listOf(
        Triple(1, 768, 768),         // single-token attention QKV
        Triple(64, 768, 768),        // attention QKV per batch
        Triple(64, 768, 3072),       // FFN expand
        Triple(64, 3072, 768),       // FFN contract
        Triple(128, 128, 128),
        Triple(256, 256, 256),
        // stage2 학습 실제 shape (8 layer × 96 emb × 3 head × headDim 32 × block 32)
        Triple(32, 96, 96),          // QKV projection
        Triple(32, 96, 256),         // SwiGLU expand
        Triple(32, 256, 96),         // SwiGLU contract
        Triple(32, 96, 2000),        // tied lm_head (vocab 2000)
        Triple(32, 32, 32),          // attention head Q·K^T
    )

    for ((m, k, n) in cases) {
        val rng = Random(7)
        val aData = FloatArray(m * k) { rng.nextFloat() - 0.5f }
        val bData = FloatArray(k * n) { rng.nextFloat() - 0.5f }

        repeat(warmupIters) {
            turboMatmul(TurboTensor(intArrayOf(m, k), aData), TurboTensor(intArrayOf(k, n), bData))
        }

        val turboNs = measureNanoTime {
            repeat(measureIters) {
                turboMatmul(TurboTensor(intArrayOf(m, k), aData), TurboTensor(intArrayOf(k, n), bData))
            }
        }
        val turboUs = turboNs / measureIters / 1000.0
        println("%-22s %12.1f".format("[$m, $k, $n]", turboUs))
    }
}
