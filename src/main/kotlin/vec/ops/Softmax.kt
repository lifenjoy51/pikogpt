package vec.ops

import vec.Tensor
import kotlin.math.exp

/**
 * Row-wise softmax. 입력 shape `[N, C]`를 받아 **각 행을 독립적으로** 확률 분포로 만든다.
 *
 *   Forward:  s_i = exp(x_i - max_row) / Σ_j exp(x_j - max_row)
 *             max 빼기는 지수 오버플로 방지용. 수학적으로 동일한 결과.
 *
 *   Backward (Jacobian form, row-wise):
 *             ∂L/∂x_i = s_i * ( ∂L/∂s_i - Σ_j s_j * ∂L/∂s_j )
 *
 * forward 결과 `s`는 backward에 필요하므로 caller가 붙잡아 둔다. 이 파일의 `softmaxBackward`는
 * `s`와 `gy`를 받아 입력 `x`의 grad를 계산한다.
 */
fun softmax(x: Tensor): Tensor {
    require(x.shape.size == 2) { "softmax는 2차원만: ${x.shape.contentToString()}" }
    val n = x.rows
    val c = x.cols
    val out = Tensor(intArrayOf(n, c))
    for (i in 0 until n) {
        // 행별 max
        var maxVal = Float.NEGATIVE_INFINITY
        for (j in 0 until c) {
            val v = x.data[i * c + j]
            if (v > maxVal) maxVal = v
        }
        // exp(x - max) 및 합
        var sumExp = 0.0f
        for (j in 0 until c) {
            val e = exp((x.data[i * c + j] - maxVal).toDouble()).toFloat()
            out.data[i * c + j] = e
            sumExp += e
        }
        // 정규화
        val inv = 1.0f / sumExp
        for (j in 0 until c) {
            out.data[i * c + j] *= inv
        }
    }
    return out
}

/**
 * Softmax의 backward. 행별 Jacobian을 사용.
 *
 * @param softmaxOut  forward에서 나온 s (shape `[N, C]`)
 * @param gyData      s에 대한 입력 기울기 ∂L/∂s (같은 크기)
 * @return            입력 x에 대한 기울기 ∂L/∂x (같은 크기)
 *
 * 공식 유도:
 *   s = softmax(x)
 *   ∂s_i/∂x_j = s_i * (δ_ij - s_j)
 *   ∂L/∂x_i  = Σ_j (∂L/∂s_j) * ∂s_j/∂x_i
 *            = s_i * ( ∂L/∂s_i - Σ_j s_j * ∂L/∂s_j )
 */
fun softmaxBackward(softmaxOut: Tensor, gyData: FloatArray): FloatArray {
    val n = softmaxOut.rows
    val c = softmaxOut.cols
    require(gyData.size == n * c)
    val dx = FloatArray(n * c)
    for (i in 0 until n) {
        // 행별 dot(s, gy)
        var dot = 0.0f
        for (j in 0 until c) {
            dot += softmaxOut.data[i * c + j] * gyData[i * c + j]
        }
        for (j in 0 until c) {
            val s = softmaxOut.data[i * c + j]
            dx[i * c + j] = s * (gyData[i * c + j] - dot)
        }
    }
    return dx
}
