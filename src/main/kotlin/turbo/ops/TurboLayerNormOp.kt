package turbo.ops

import turbo.TurboTensor
import kotlin.math.sqrt

/**
 * Row-wise LayerNorm. Phase 0은 vec와 동일 (Phase 1에서 RMSNorm 추가).
 *
 *   y = γ · (x - μ)/√(σ² + ε) + β
 */
class TurboLayerNormCache(
    val xHat: FloatArray,
    val invStd: FloatArray,
)

fun turboLayerNormForward(
    x: TurboTensor,
    gamma: TurboTensor,
    beta: TurboTensor,
    eps: Float = 1e-5f,
): Pair<TurboTensor, TurboLayerNormCache> {
    require(x.shape.size == 2)
    val n = x.rows
    val c = x.cols
    require(gamma.numel == c && beta.numel == c) {
        "γ,β 차원은 마지막 축과 일치해야 함"
    }

    val y = TurboTensor(intArrayOf(n, c))
    val xHat = FloatArray(n * c)
    val invStd = FloatArray(n)

    for (i in 0 until n) {
        var mean = 0.0f
        for (j in 0 until c) mean += x.data[i * c + j]
        mean /= c
        var variance = 0.0f
        for (j in 0 until c) {
            val d = x.data[i * c + j] - mean
            variance += d * d
        }
        variance /= c
        val inv = 1.0f / sqrt(variance + eps)
        invStd[i] = inv
        for (j in 0 until c) {
            val h = (x.data[i * c + j] - mean) * inv
            xHat[i * c + j] = h
            y.data[i * c + j] = gamma.data[j] * h + beta.data[j]
        }
    }
    return y to TurboLayerNormCache(xHat, invStd)
}

fun turboLayerNormBackward(
    cache: TurboLayerNormCache,
    gamma: TurboTensor,
    beta: TurboTensor,
    gyData: FloatArray,
    rows: Int,
    cols: Int,
): FloatArray {
    val dGamma = gamma.gradOrAlloc()
    val dBeta = beta.gradOrAlloc()
    val dx = FloatArray(rows * cols)

    for (i in 0 until rows) {
        for (j in 0 until cols) {
            dGamma[j] += gyData[i * cols + j] * cache.xHat[i * cols + j]
            dBeta[j] += gyData[i * cols + j]
        }
        var meanDxHat = 0.0f
        var meanDxHatXHat = 0.0f
        val invC = 1.0f / cols
        for (j in 0 until cols) {
            val dxHat = gyData[i * cols + j] * gamma.data[j]
            meanDxHat += dxHat
            meanDxHatXHat += dxHat * cache.xHat[i * cols + j]
        }
        meanDxHat *= invC
        meanDxHatXHat *= invC

        val inv = cache.invStd[i]
        for (j in 0 until cols) {
            val dxHat = gyData[i * cols + j] * gamma.data[j]
            dx[i * cols + j] = inv * (dxHat - meanDxHat - cache.xHat[i * cols + j] * meanDxHatXHat)
        }
    }
    return dx
}
