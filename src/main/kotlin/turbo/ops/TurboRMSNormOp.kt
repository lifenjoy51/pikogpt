package turbo.ops

import turbo.TurboTensor
import kotlin.math.sqrt

/**
 * Root Mean Square Normalization (Llama 표준).
 *
 *   Forward:  rms_i  = √(mean(x²) + eps)
 *             y_ij   = (x_ij / rms_i) * γ_j
 *
 *   LayerNorm 대비 중심화(평균 0) 단계와 β가 없다 → 파라미터 절반, ~7% 빠름.
 *
 *   Backward (행별, c = invRms = 1/rms):
 *     a_j = gy_j * γ_j
 *     b   = mean(x_j * a_j)         (행 평균)
 *     dL/dx_k  = c * a_k - c³ * b * x_k
 *     dL/dγ_j += Σ_i gy_ij * x_ij * c_i
 */
class TurboRMSNormCache(
    /** backward에 필요한 원본 x 데이터 (입력은 보존 안 되므로 복사). */
    val xData: FloatArray,
    /** 행별 invRms = 1/√(mean(x²)+eps). */
    val invRms: FloatArray,
)

fun turboRmsNormForward(
    x: TurboTensor,
    gamma: TurboTensor,
    eps: Float = 1e-5f,
): Pair<TurboTensor, TurboRMSNormCache> {
    require(x.shape.size == 2)
    val n = x.rows
    val c = x.cols
    require(gamma.numel == c) { "γ 차원은 마지막 축과 일치해야 함" }

    val y = TurboTensor(intArrayOf(n, c))
    val invRms = FloatArray(n)
    val invC = 1.0f / c

    for (i in 0 until n) {
        var sumSq = 0.0f
        for (j in 0 until c) {
            val v = x.data[i * c + j]
            sumSq += v * v
        }
        val ms = sumSq * invC
        val inv = 1.0f / sqrt(ms + eps)
        invRms[i] = inv
        for (j in 0 until c) {
            y.data[i * c + j] = x.data[i * c + j] * inv * gamma.data[j]
        }
    }
    return y to TurboRMSNormCache(x.data.copyOf(), invRms)
}

fun turboRmsNormBackward(
    cache: TurboRMSNormCache,
    gamma: TurboTensor,
    gyData: FloatArray,
    rows: Int,
    cols: Int,
): FloatArray {
    val dGamma = gamma.gradOrAlloc()
    val dx = FloatArray(rows * cols)
    val invC = 1.0f / cols

    for (i in 0 until rows) {
        val invRms = cache.invRms[i]
        val invRms3 = invRms * invRms * invRms

        var meanXa = 0.0f
        for (j in 0 until cols) {
            val aj = gyData[i * cols + j] * gamma.data[j]
            meanXa += cache.xData[i * cols + j] * aj
            dGamma[j] += gyData[i * cols + j] * cache.xData[i * cols + j] * invRms
        }
        meanXa *= invC

        for (j in 0 until cols) {
            val aj = gyData[i * cols + j] * gamma.data[j]
            dx[i * cols + j] = invRms * aj - invRms3 * meanXa * cache.xData[i * cols + j]
        }
    }
    return dx
}
