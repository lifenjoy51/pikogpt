package vec.layer

import vec.Tensor
import vec.ops.gelu
import vec.ops.geluBackward

/**
 * Transformer FFN 블록: expand → GELU → contract → dropout.
 *
 *   x [N, C] --fc--> h [N, 4C] --gelu--> a [N, 4C] --proj--> p [N, C] --dropout--> y [N, C]
 *
 *   Forward:  h = fc(x);  a = gelu(h);  p = proj(a);  y = dropout(p)
 *   Backward: 체인 룰을 그대로
 *     dp = dropout.backward(gy)           → ∂L/∂p
 *     da = proj.backward(dp)              → ∂L/∂a
 *     dh = geluBackward(cachedH, da)      → ∂L/∂h
 *     dx = fc.backward(Tensor(...,dh))    → ∂L/∂x
 *
 * 각 하위 레이어가 자체 param grad를 누적하므로 여기선 따로 관여할 일 없음.
 * dropout은 파라미터가 없으므로 parameters()에 기여 없음.
 */
class MLP(val embedDim: Int, useBias: Boolean = true, dropoutProbability: Float = 0.0f) {
    val fullyConnected: Linear = Linear(embedDim, 4 * embedDim, useBias)
    val projection: Linear = Linear(4 * embedDim, embedDim, useBias)
    val dropout: Dropout = Dropout(dropoutProbability)

    /** backward에서 GELU의 입력(h = fc(x))을 재사용. */
    private var cachedH: Tensor? = null

    fun forward(x: Tensor): Tensor {
        val h = fullyConnected.forward(x)
        cachedH = h
        val a = gelu(h)
        val p = projection.forward(a)
        return dropout.forward(p)
    }

    fun backward(gy: Tensor): Tensor {
        val h = cachedH ?: error("forward 없이 backward 호출")
        val dp = dropout.backward(gy)                                    // [N, C]
        val da = projection.backward(dp)                                 // [N, 4C]
        val dhData = geluBackward(h, da.data)                            // [N, 4C]
        val dh = Tensor(intArrayOf(da.rows, da.cols), dhData)
        return fullyConnected.backward(dh)                               // [N, C]
    }

    fun parameters(): List<Tensor> = fullyConnected.parameters() + projection.parameters()
}
