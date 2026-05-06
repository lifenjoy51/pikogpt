package turbo.ops

import turbo.TurboTensor
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin

/**
 * Rotary Position Embedding (GPT-NeoX half-split). Phase 0은 vec와 동일.
 * Phase 2에서 cos/sin 테이블 캐시 + SIMD 회전으로 교체.
 *
 *   theta_i = 10000^(-2i/D),  angle = pos * theta_i
 *   x'[i]      = x[i] * cos - x[i+D/2] * sin
 *   x'[i+D/2]  = x[i] * sin + x[i+D/2] * cos
 */
fun turboApplyRoPE(x: TurboTensor, numHeads: Int) {
    require(x.shape.size == 2) { "RoPE 입력은 [T, H*D] 형태" }
    val t = x.rows
    val totalDim = x.cols
    val headDim = totalDim / numHeads
    require(totalDim % numHeads == 0) { "embedDim must be divisible by numHeads" }
    require(headDim % 2 == 0) { "head_dim must be even for RoPE (got $headDim)" }
    val half = headDim / 2

    for (pos in 0 until t) {
        for (h in 0 until numHeads) {
            val headOffset = h * headDim
            for (i in 0 until half) {
                val theta = 10000.0.pow(-2.0 * i / headDim)
                val angle = pos * theta
                val c = cos(angle).toFloat()
                val s = sin(angle).toFloat()

                val idxA = pos * totalDim + headOffset + i
                val idxB = pos * totalDim + headOffset + i + half
                val xa = x.data[idxA]
                val xb = x.data[idxB]
                x.data[idxA] = xa * c - xb * s
                x.data[idxB] = xa * s + xb * c
            }
        }
    }
}

fun turboApplyRoPEBackward(dx: TurboTensor, numHeads: Int) {
    require(dx.shape.size == 2)
    val t = dx.rows
    val totalDim = dx.cols
    val headDim = totalDim / numHeads
    require(headDim % 2 == 0)
    val half = headDim / 2

    for (pos in 0 until t) {
        for (h in 0 until numHeads) {
            val headOffset = h * headDim
            for (i in 0 until half) {
                val theta = 10000.0.pow(-2.0 * i / headDim)
                val angle = pos * theta
                val c = cos(angle).toFloat()
                val s = sin(angle).toFloat()

                val idxA = pos * totalDim + headOffset + i
                val idxB = pos * totalDim + headOffset + i + half
                val da = dx.data[idxA]
                val db = dx.data[idxB]
                dx.data[idxA] = da * c + db * s
                dx.data[idxB] = -da * s + db * c
            }
        }
    }
}
