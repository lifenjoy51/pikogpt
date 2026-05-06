package turbo.ops

import turbo.TurboTensor
import kotlin.math.exp

/**
 * Row-wise softmax. Phase 0은 vec와 동일 (max-shift + exp + normalize).
 * Phase 2에서 SIMD exp/div, Phase 3에서 Flash Attention용 online softmax 추가 예정.
 */
fun turboSoftmax(x: TurboTensor): TurboTensor {
    require(x.shape.size == 2) { "turboSoftmax는 2차원만: ${x.shape.contentToString()}" }
    val n = x.rows
    val c = x.cols
    val out = TurboTensor(intArrayOf(n, c))
    for (i in 0 until n) {
        var maxVal = Float.NEGATIVE_INFINITY
        for (j in 0 until c) {
            val v = x.data[i * c + j]
            if (v > maxVal) maxVal = v
        }
        var sumExp = 0.0f
        for (j in 0 until c) {
            val e = exp((x.data[i * c + j] - maxVal).toDouble()).toFloat()
            out.data[i * c + j] = e
            sumExp += e
        }
        val inv = 1.0f / sumExp
        for (j in 0 until c) {
            out.data[i * c + j] *= inv
        }
    }
    return out
}

fun turboSoftmaxBackward(softmaxOut: TurboTensor, gyData: FloatArray): FloatArray {
    val n = softmaxOut.rows
    val c = softmaxOut.cols
    require(gyData.size == n * c)
    val dx = FloatArray(n * c)
    for (i in 0 until n) {
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
