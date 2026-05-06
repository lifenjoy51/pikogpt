package turbo.ops

import turbo.TurboTensor
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin

/**
 * RoPE를 단일 토큰 (rows=1) + 명시적 position에 적용 — KV cache incremental decode 용.
 *
 * 일반 turboApplyRoPE는 rows=T를 받아 row index를 position으로 사용. incremental 모드에서는
 * 토큰 한 개의 position이 cache.currentPosition이므로 별도 함수.
 */
fun turboApplyRoPEAtPosition(x: TurboTensor, numHeads: Int, position: Int) {
    require(x.shape.size == 2) { "RoPE 입력은 [T, H*D] 형태" }
    require(x.rows == 1) { "turboApplyRoPEAtPosition은 단일 토큰만 (rows=1)" }
    val totalDim = x.cols
    val headDim = totalDim / numHeads
    require(headDim % 2 == 0) { "head_dim must be even" }
    val half = headDim / 2

    for (h in 0 until numHeads) {
        val headOffset = h * headDim
        for (i in 0 until half) {
            val theta = 10000.0.pow(-2.0 * i / headDim)
            val angle = position * theta
            val c = cos(angle).toFloat()
            val s = sin(angle).toFloat()
            val xa = x.data[headOffset + i]
            val xb = x.data[headOffset + i + half]
            x.data[headOffset + i] = xa * c - xb * s
            x.data[headOffset + i + half] = xa * s + xb * c
        }
    }
}
