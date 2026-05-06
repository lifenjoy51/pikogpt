package turbo.layer

import turbo.TurboTensor
import turbo.ops.TurboLayerNormCache
import turbo.ops.turboLayerNormBackward
import turbo.ops.turboLayerNormForward
import turbo.turboTensorOnes
import turbo.turboTensorZeros

/**
 * Layer Normalization 레이어. Phase 1에서 TurboNorm sealed interface 구현.
 */
class TurboLayerNorm(
    val dim: Int,
    val useBias: Boolean = true,
    val eps: Float = 1e-5f,
) : TurboNorm {
    val gamma: TurboTensor = turboTensorOnes(intArrayOf(dim))
    val beta: TurboTensor = if (useBias) turboTensorZeros(intArrayOf(dim)) else turboTensorZeros(intArrayOf(dim))

    private var cache: TurboLayerNormCache? = null
    private var inputRows: Int = 0
    private var inputCols: Int = 0

    override fun forward(x: TurboTensor): TurboTensor {
        val (y, c) = turboLayerNormForward(x, gamma, beta, eps)
        cache = c
        inputRows = x.rows
        inputCols = x.cols
        return y
    }

    override fun backward(gy: TurboTensor): TurboTensor {
        val c = cache ?: error("forward 없이 backward 호출")
        val dxData = turboLayerNormBackward(c, gamma, beta, gy.data, inputRows, inputCols)
        return TurboTensor(intArrayOf(inputRows, inputCols), dxData)
    }

    override fun parameters(): List<TurboTensor> = if (useBias) listOf(gamma, beta) else listOf(gamma)
}
