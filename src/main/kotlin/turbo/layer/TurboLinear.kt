package turbo.layer

import jdk.incubator.vector.FloatVector
import turbo.TurboSimdMath
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

        // dW[o, i] += Σ_n gy[n, o] * x[n, i]   — n outer, i inner SIMD scaled-add
        val wGrad = weight.gradOrAlloc()
        val species = TurboSimdMath.SPECIES
        val iUpper = species.loopBound(inFeatures)
        val gyData = gy.data
        val xData = x.data
        val rows = gy.rows
        for (n in 0 until rows) {
            val gyOff = n * outFeatures
            val xOff = n * inFeatures
            for (o in 0 until outFeatures) {
                val gyno = gyData[gyOff + o]
                if (gyno == 0.0f) continue
                val vScalar = FloatVector.broadcast(species, gyno)
                val wOff = o * inFeatures
                var i = 0
                while (i < iUpper) {
                    val vX = FloatVector.fromArray(species, xData, xOff + i)
                    val vW = FloatVector.fromArray(species, wGrad, wOff + i)
                    vX.fma(vScalar, vW).intoArray(wGrad, wOff + i)
                    i += species.length()
                }
                while (i < inFeatures) {
                    wGrad[wOff + i] += gyno * xData[xOff + i]
                    i++
                }
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
