package vec.layer

import vec.Tensor
import vec.ops.matmul
import vec.ops.matmulBackward
import vec.tensorGaussian
import vec.tensorZeros
import vec.transpose2D

/**
 * 선형 변환 레이어. 스칼라 `gpt.Linear`의 벡터 버전.
 *
 *   Shape: x[N, inF] · W^T[inF, outF] + b[outF] = y[N, outF]
 *   (W 자체는 [outF, inF] 형태로 저장하여 torch 관행과 일치)
 *
 *   Forward:   y = x · W^T + b
 *   Backward (chain rule):
 *     ∂L/∂x = ∂L/∂y · W                    (shape [N, inF])
 *     ∂L/∂W += ∂L/∂y^T · x                 (shape [outF, inF])
 *     ∂L/∂b += Σ_rows(∂L/∂y)               (shape [outF])
 *
 * backward 함수는 **입력 기울기를 반환**하고 weight/bias grad는 각 Tensor의 grad에
 * **누적**한다. optimizer가 step 전에 `zeroGrad`를 호출해 리셋.
 */
class Linear(
    val inFeatures: Int,
    val outFeatures: Int,
    val useBias: Boolean = true,
) {
    val weight: Tensor = tensorGaussian(intArrayOf(outFeatures, inFeatures), std = 0.02f)
    val bias: Tensor? = if (useBias) tensorZeros(intArrayOf(outFeatures)) else null

    /** backward에서 쓰기 위한 입력 캐시. */
    private var cachedInput: Tensor? = null

    fun forward(x: Tensor): Tensor {
        require(x.shape.size == 2 && x.cols == inFeatures) {
            "Linear 입력 shape 불일치: ${x.shape.contentToString()} vs inF=$inFeatures"
        }
        cachedInput = x
        val y = matmul(x, weight.transpose2D())  // [N, outF]
        if (bias != null) {
            // row-wise broadcast: y[i, j] += b[j]
            for (i in 0 until y.rows) {
                for (j in 0 until y.cols) {
                    y.data[i * y.cols + j] += bias.data[j]
                }
            }
        }
        return y
    }

    fun backward(gy: Tensor): Tensor {
        val x = cachedInput ?: error("forward 없이 backward 호출")
        require(gy.rows == x.rows && gy.cols == outFeatures)

        // ∂L/∂W += gy^T · x     — weight 자체 shape [outF, inF]
        //   matmulBackward는 (gy · W) 형태의 입력을 가정하므로
        //   여기서는 수동으로 계산해 weight.grad에 누적.
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

        // ∂L/∂b += Σ_rows gy
        if (bias != null) {
            val bGrad = bias.gradOrAlloc()
            for (n in 0 until gy.rows) {
                for (o in 0 until outFeatures) {
                    bGrad[o] += gy.data[n * outFeatures + o]
                }
            }
        }

        // ∂L/∂x = gy · W        — matmul(gy[N,outF], W[outF,inF]) → [N, inF]
        return matmul(gy, weight)
    }

    fun parameters(): List<Tensor> = listOfNotNull(weight, bias)

    /** 간접적으로 matmulBackward 패턴을 재사용하기 싫어서 수동 계산했음을 드러내는 주석. */
    @Suppress("unused")
    private fun reuseMatmulBackwardAlternative(gy: Tensor, x: Tensor) {
        // 대안: matmulBackward를 호출하려면 y = x · W^T 형태로 보고
        // (a=x, b=W^T) 에 대한 backward를 실행하면 W^T의 grad가 나옴.
        // 그러나 weight는 W로 저장되어 있어 전치 역연산을 다시 해야 한다.
        // 직접 루프가 오히려 간단해서 위 구현을 선택.
        matmulBackward(x, weight.transpose2D(), gy.data)
    }
}
