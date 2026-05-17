package turbo.bench

import mps.MpsAvailability
import mps.ops.mpsMatmul
import turbo.TurboSimdMath
import turbo.TurboTensor
import turbo.ops.turboMatmul
import kotlin.random.Random
import kotlin.system.measureNanoTime

/**
 * turbo CPU SIMD vs mps Metal GPU per-call us 비교.
 *
 * 10M 모델 (`Bench10MTurbo`: d=256 L=16 H=8 blockSize=32 batch=2) 실제 matmul shape 위주.
 *   - QKV/output projection: [64, 256] · [256, 256]
 *   - SwiGLU expand:         [64, 256] · [256, 1024]
 *   - SwiGLU contract:       [64, 1024] · [1024, 256]
 *   - LM head tied:          [64, 256] · [256, 2000]
 *   - Q·K^T per head:        [32, 32, 32]
 *
 * 첫 호출 init cost(약 1초) 분리를 위해 warmup 20회 + 측정 100회.
 */
fun main() {
    println("=== Turbo CPU SIMD vs MPS Metal GPU MatMul micro-bench ===")
    println("SIMD lane count: ${TurboSimdMath.laneCount} (FloatVector.SPECIES_PREFERRED)")
    println("CPU cores: ${Runtime.getRuntime().availableProcessors()}")

    val mpsOk = MpsAvailability.ensureChecked()
    println("MPS available: $mpsOk (${MpsAvailability.reason})")
    println()

    println(
        "%-24s %12s %12s %10s".format(
            "shape", "turbo (us)", "mps (us)", "speedup"
        )
    )
    println("-".repeat(64))

    // 10M 모델 실제 shape + 기준점 비교용 일반 shape.
    val cases = listOf(
        // 10M model (d=256, B*T=64, vocab=2000, MLP 4x expand)
        Triple(64, 256, 256),    // QKV/output proj
        Triple(64, 256, 1024),   // SwiGLU expand
        Triple(64, 1024, 256),   // SwiGLU contract
        Triple(64, 256, 2000),   // LM head (tied)
        // attention internal
        Triple(32, 32, 32),      // Q·K^T per head (T=32, headDim=32)
        // 더 큰 shape — GPU 효율 가시화
        Triple(128, 256, 256),
        Triple(256, 512, 512),
        Triple(64, 768, 3072),   // GPT-2 small FFN expand 비교점
        Triple(64, 3072, 768),
    )

    val warmupIters = 20
    val measureIters = 100

    for ((m, k, n) in cases) {
        val rng = Random((m * 131L + k * 17L + n))
        val aData = FloatArray(m * k) { rng.nextFloat() - 0.5f }
        val bData = FloatArray(k * n) { rng.nextFloat() - 0.5f }

        // turbo warmup + 측정
        repeat(warmupIters) {
            turboMatmul(TurboTensor(intArrayOf(m, k), aData), TurboTensor(intArrayOf(k, n), bData))
        }
        val turboNs = measureNanoTime {
            repeat(measureIters) {
                turboMatmul(TurboTensor(intArrayOf(m, k), aData), TurboTensor(intArrayOf(k, n), bData))
            }
        }
        val turboUs = turboNs / measureIters / 1000.0

        // mps warmup + 측정 (가능 시)
        val (mpsUs, speedup) = if (mpsOk) {
            repeat(warmupIters) {
                mpsMatmul(TurboTensor(intArrayOf(m, k), aData), TurboTensor(intArrayOf(k, n), bData))
            }
            val mpsNs = measureNanoTime {
                repeat(measureIters) {
                    mpsMatmul(TurboTensor(intArrayOf(m, k), aData), TurboTensor(intArrayOf(k, n), bData))
                }
            }
            val us = mpsNs / measureIters / 1000.0
            us to (turboUs / us)
        } else {
            Double.NaN to Double.NaN
        }

        if (mpsUs.isNaN()) {
            println("%-24s %12.1f %12s %10s".format("[$m, $k, $n]", turboUs, "-", "-"))
        } else {
            println("%-24s %12.1f %12.1f %9.2fx".format("[$m, $k, $n]", turboUs, mpsUs, speedup))
        }
    }
}
