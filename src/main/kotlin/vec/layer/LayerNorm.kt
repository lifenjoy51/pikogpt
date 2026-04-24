package vec.layer

import vec.Tensor
import vec.ops.LayerNormCache
import vec.ops.layerNormBackward
import vec.ops.layerNormForward
import vec.tensorOnes
import vec.tensorZeros

/**
 * Layer Normalization 레이어. γ(초깃값 1)와 β(초깃값 0)을 파라미터로 보유하며
 * 실제 수식은 `vec.ops.LayerNormOp`에 위임한다. 이 레이어는:
 *   - 파라미터 소유·초기화
 *   - forward 결과에 필요한 cache(xHat, invStd) 관리
 *   - backward 시 캐시를 재사용해 γ/β/x grad 계산
 * 을 담당한다.
 *
 * 수식/기울기 유도는 `LayerNormOp.kt`의 주석 참고.
 */
class LayerNorm(
    val dim: Int,
    val useBias: Boolean = true,
    val eps: Float = 1e-5f,
) {
    val gamma: Tensor = tensorOnes(intArrayOf(dim))
    val beta: Tensor = if (useBias) tensorZeros(intArrayOf(dim)) else tensorZeros(intArrayOf(dim))

    private var cache: LayerNormCache? = null
    private var inputRows: Int = 0
    private var inputCols: Int = 0

    fun forward(x: Tensor): Tensor {
        val (y, c) = layerNormForward(x, gamma, beta, eps)
        cache = c
        inputRows = x.rows
        inputCols = x.cols
        return y
    }

    fun backward(gy: Tensor): Tensor {
        val c = cache ?: error("forward 없이 backward 호출")
        val dxData = layerNormBackward(c, gamma, beta, gy.data, inputRows, inputCols)
        // dx를 Tensor로 감싸 반환 (shape = [N, C])
        return Tensor(intArrayOf(inputRows, inputCols), dxData)
    }

    fun parameters(): List<Tensor> = if (useBias) listOf(gamma, beta) else listOf(gamma)
}
