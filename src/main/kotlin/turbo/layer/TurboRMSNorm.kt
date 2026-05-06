package turbo.layer

import turbo.TurboTensor
import turbo.ops.TurboRMSNormCache
import turbo.ops.turboRmsNormBackward
import turbo.ops.turboRmsNormForward
import turbo.turboTensorOnes

/**
 * RMSNorm 레이어. γ만 학습 (β 없음). LayerNorm 대비 파라미터 절반.
 */
class TurboRMSNorm(
    val dim: Int,
    val eps: Float = 1e-5f,
) : TurboNorm {
    val gamma: TurboTensor = turboTensorOnes(intArrayOf(dim))

    private var cache: TurboRMSNormCache? = null
    private var inputRows: Int = 0
    private var inputCols: Int = 0

    override fun forward(x: TurboTensor): TurboTensor {
        val (y, c) = turboRmsNormForward(x, gamma, eps)
        cache = c
        inputRows = x.rows
        inputCols = x.cols
        return y
    }

    override fun backward(gy: TurboTensor): TurboTensor {
        val c = cache ?: error("forward 없이 backward 호출")
        val dxData = turboRmsNormBackward(c, gamma, gy.data, inputRows, inputCols)
        return TurboTensor(intArrayOf(inputRows, inputCols), dxData)
    }

    override fun parameters(): List<TurboTensor> = listOf(gamma)
}
