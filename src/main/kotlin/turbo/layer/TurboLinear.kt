package turbo.layer

import turbo.TurboTensor
import turbo.ops.turboMatmul
import turbo.transpose2D
import turbo.turboTensorGaussian
import turbo.turboTensorZeros

/**
 * 선형 변환 레이어. Phase 0은 vec.layer.VecLinear와 동일.
 *   y = x · W^T + b,  W shape [outF, inF]
 *
 * Phase 2에서 weight pre-transpose 캐시 + SIMD matmul로 교체.
 */
class TurboLinear(
    val inFeatures: Int,
    val outFeatures: Int,
    val useBias: Boolean = true,
) {
    val weight: TurboTensor = turboTensorGaussian(intArrayOf(outFeatures, inFeatures), std = 0.02f)
    val bias: TurboTensor? = if (useBias) turboTensorZeros(intArrayOf(outFeatures)) else null

    private var cachedInput: TurboTensor? = null

    fun forward(x: TurboTensor): TurboTensor {
        require(x.shape.size == 2 && x.cols == inFeatures) {
            "TurboLinear 입력 shape 불일치: ${x.shape.contentToString()} vs inF=$inFeatures"
        }
        cachedInput = x
        val y = turboMatmul(x, weight.transpose2D())
        if (bias != null) {
            for (i in 0 until y.rows) {
                for (j in 0 until y.cols) {
                    y.data[i * y.cols + j] += bias.data[j]
                }
            }
        }
        return y
    }

    fun backward(gy: TurboTensor): TurboTensor {
        val x = cachedInput ?: error("forward 없이 backward 호출")
        require(gy.rows == x.rows && gy.cols == outFeatures)

        val wGrad = weight.gradOrAlloc()
        for (o in 0 until outFeatures) {
            for (i in 0 until inFeatures) {
                var sum = 0.0f
                for (n in 0 until gy.rows) {
                    sum += gy.data[n * outFeatures + o] * x.data[n * inFeatures + i]
                }
                wGrad[o * inFeatures + i] += sum
            }
        }

        if (bias != null) {
            val bGrad = bias.gradOrAlloc()
            for (n in 0 until gy.rows) {
                for (o in 0 until outFeatures) {
                    bGrad[o] += gy.data[n * outFeatures + o]
                }
            }
        }

        return turboMatmul(gy, weight)
    }

    fun parameters(): List<TurboTensor> = listOfNotNull(weight, bias)
}
