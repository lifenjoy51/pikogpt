package turbo.bench

import turbo.TurboSimdMath
import turbo.TurboTensor
import turbo.ops.turboMatmul
import vec.Tensor as VecTensor
import vec.ops.matmul as vecMatmul
import kotlin.random.Random
import kotlin.system.measureNanoTime

/**
 * Phase 2 마이크로벤치마크 — vec(naive ikj) vs turbo(SIMD blocked) MatMul 비교.
 *
 * 결과는 환경마다 다르지만 Apple Silicon NEON (lane 4) 기준 1M 모델 hot shapes에서
 * **2~4×** 가속이 기대치. AVX2/AVX-512에서 더 큼.
 */
fun main() {
    println("=== Turbo MatMul / AdamW micro-benchmark ===")
    println("SIMD lane count: ${TurboSimdMath.laneCount} (FloatVector.SPECIES_PREFERRED)")
    println("CPU cores: ${Runtime.getRuntime().availableProcessors()}")
    println()

    benchMatmul()
}

private fun benchMatmul() {
    println("--- MatMul forward ---")
    println("%-22s %12s %12s %10s".format("shape", "turbo (us)", "vec (us)", "speedup"))
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
            vecMatmul(VecTensor(intArrayOf(m, k), aData), VecTensor(intArrayOf(k, n), bData))
        }

        val turboNs = measureNanoTime {
            repeat(measureIters) {
                turboMatmul(TurboTensor(intArrayOf(m, k), aData), TurboTensor(intArrayOf(k, n), bData))
            }
        }
        val vecNs = measureNanoTime {
            repeat(measureIters) {
                vecMatmul(VecTensor(intArrayOf(m, k), aData), VecTensor(intArrayOf(k, n), bData))
            }
        }
        val turboUs = turboNs / measureIters / 1000.0
        val vecUs = vecNs / measureIters / 1000.0
        val speedup = vecUs / turboUs
        println("%-22s %12.1f %12.1f %9.2fx".format("[$m, $k, $n]", turboUs, vecUs, speedup))
    }
}
