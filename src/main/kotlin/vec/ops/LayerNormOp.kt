package vec.ops

import vec.Tensor
import kotlin.math.sqrt

/**
 * Row-wise Layer Normalization.
 *
 * 입력 shape `[N, C]`. 각 행을 평균 0 / 분산 1로 정규화한 뒤 학습 가능한 γ, β로 affine.
 *
 *   μ       = mean over C
 *   σ²      = var over C
 *   x_hat   = (x - μ) / √(σ² + eps)
 *   y       = γ * x_hat + β
 *
 * Backward: 행별로 다음이 성립 (유도 표준, https://arxiv.org/abs/1607.06450 부록 참고).
 *
 *   dL/dγ += Σ_rows (dL/dy * x_hat)
 *   dL/dβ += Σ_rows (dL/dy)
 *   let  dxHat = dL/dy * γ   (행별, 원소별)
 *   let  inv   = 1 / √(σ² + eps)
 *   dL/dx = inv * ( dxHat - mean(dxHat) - x_hat * mean(dxHat * x_hat) )
 *
 * forward 결과 `x_hat`과 `inv`를 backward가 재사용하므로 별도 캐시로 반환한다.
 */
class LayerNormCache(
    val xHat: FloatArray,   // shape N*C
    val invStd: FloatArray, // shape N
)

fun layerNormForward(
    x: Tensor,
    gamma: Tensor,
    beta: Tensor,
    eps: Float = 1e-5f,
): Pair<Tensor, LayerNormCache> {
    require(x.shape.size == 2)
    val n = x.rows
    val c = x.cols
    require(gamma.numel == c && beta.numel == c) {
        "γ,β 차원은 마지막 축과 일치해야 함"
    }

    val y = Tensor(intArrayOf(n, c))
    val xHat = FloatArray(n * c)
    val invStd = FloatArray(n)

    for (i in 0 until n) {
        // 평균
        var mean = 0.0f
        for (j in 0 until c) mean += x.data[i * c + j]
        mean /= c
        // 분산
        var variance = 0.0f
        for (j in 0 until c) {
            val d = x.data[i * c + j] - mean
            variance += d * d
        }
        variance /= c
        val inv = 1.0f / sqrt(variance + eps)
        invStd[i] = inv
        // 정규화 + affine
        for (j in 0 until c) {
            val h = (x.data[i * c + j] - mean) * inv
            xHat[i * c + j] = h
            y.data[i * c + j] = gamma.data[j] * h + beta.data[j]
        }
    }
    return y to LayerNormCache(xHat, invStd)
}

/**
 * LayerNorm backward. γ/β의 grad에 누적하고 x의 grad(반환값)를 계산한다.
 */
fun layerNormBackward(
    cache: LayerNormCache,
    gamma: Tensor,
    beta: Tensor,
    gyData: FloatArray,
    rows: Int,
    cols: Int,
): FloatArray {
    val dGamma = gamma.gradOrAlloc()
    val dBeta = beta.gradOrAlloc()
    val dx = FloatArray(rows * cols)

    for (i in 0 until rows) {
        // dγ, dβ 누적
        for (j in 0 until cols) {
            dGamma[j] += gyData[i * cols + j] * cache.xHat[i * cols + j]
            dBeta[j] += gyData[i * cols + j]
        }
        // dxHat = gy * γ (행별)
        // 그리고 두 평균: mean(dxHat), mean(dxHat * xHat)
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
