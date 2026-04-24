package vec.layer

import vec.Tensor

/**
 * Pre-LayerNorm Transformer 블록.
 *
 *   Forward:
 *     h1 = x + attention(layerNorm1(x))
 *     y  = h1 + mlp(layerNorm2(h1))
 *
 *   Residual 연결 덕분에 backward도 "잔차(skip) 기여 + 하위 레이어 기여"로 깔끔.
 *
 *   Backward (체인 룰):
 *     y = h1 + mlp(ln2(h1))
 *       ⇒ ∂L/∂mlp_out = ∂L/∂y
 *         ∂L/∂h1 = ∂L/∂y  +  ln2.backward( mlp.backward(∂L/∂y) )
 *
 *     h1 = x + attention(ln1(x))
 *       ⇒ ∂L/∂attn_out = ∂L/∂h1
 *         ∂L/∂x = ∂L/∂h1  +  ln1.backward( attention.backward(∂L/∂h1) )
 *
 *   (잔차 경로는 identity라 grad가 그대로 통과, 합은 원소별.)
 */
class TransformerBlock(
    val embedDim: Int,
    val numHeads: Int,
    useBias: Boolean = true,
) {
    val layerNorm1: LayerNorm = LayerNorm(embedDim, useBias)
    val attention: SelfAttention = SelfAttention(embedDim, numHeads, useBias)
    val layerNorm2: LayerNorm = LayerNorm(embedDim, useBias)
    val mlp: MLP = MLP(embedDim, useBias)

    fun forward(x: Tensor): Tensor {
        val attnOut = attention.forward(layerNorm1.forward(x))
        val h1 = addElementwise(x, attnOut)                    // h1 = x + attn(ln1(x))
        val mlpOut = mlp.forward(layerNorm2.forward(h1))
        return addElementwise(h1, mlpOut)                      // y = h1 + mlp(ln2(h1))
    }

    fun backward(gy: Tensor): Tensor {
        // MLP/LN2 분기: ∂L/∂h1_branch = ln2.backward( mlp.backward(gy) )
        val dLn2Out = mlp.backward(gy)                         // ∂L/∂ln2_out
        val dH1FromMlp = layerNorm2.backward(dLn2Out)          // ∂L/∂h1 (branch)
        val dH1 = addElementwise(gy, dH1FromMlp)               // residual gy + branch

        // Attention/LN1 분기: ∂L/∂x_branch = ln1.backward( attention.backward(dH1) )
        val dLn1Out = attention.backward(dH1)
        val dXFromAttn = layerNorm1.backward(dLn1Out)
        return addElementwise(dH1, dXFromAttn)                 // ∂L/∂x = dH1 + branch
    }

    fun parameters(): List<Tensor> =
        layerNorm1.parameters() +
                attention.parameters() +
                layerNorm2.parameters() +
                mlp.parameters()

    /** 두 같은 shape 텐서를 원소별로 더해 새 Tensor 반환. */
    private fun addElementwise(a: Tensor, b: Tensor): Tensor {
        require(a.numel == b.numel)
        val out = Tensor(a.shape.copyOf())
        for (i in out.data.indices) out.data[i] = a.data[i] + b.data[i]
        return out
    }
}
