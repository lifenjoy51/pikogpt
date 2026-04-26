package vec.layer

import vec.Tensor
import vec.ops.gelu
import vec.ops.geluBackward
import vec.ops.silu
import vec.ops.siluBackward

/**
 * Transformer FFN 블록. activation 종류에 따라 두 변형 지원:
 *
 * **GELU (default, GPT-2/3 스타일)**:
 *   x [N, C] --fc--> h [N, 4C] --gelu--> a [N, 4C] --proj--> p [N, C] --dropout--> y [N, C]
 *   - fullyConnected (= up): C → 4C
 *   - projection (= down):  4C → C
 *
 * **SwiGLU (Llama 스타일)**:
 *   x [N, C] -- gate_proj --> g [N, H]
 *           -- up_proj   --> u [N, H]
 *   h = silu(g) ⊙ u  (element-wise)
 *   p = down_proj(h) [N, C]
 *   y = dropout(p)
 *   - hidden_dim H = (8/3) * C 반올림 (Llama 공식, GELU 4C 대비 ~동일 params)
 *   - 3개 Linear 사용 (gate, up, down) — params는 GELU와 거의 같음
 *   - SwiGLU(x) = SiLU(W_gate x) ⊙ (W_up x), 그리고 W_down 적용
 *
 * 둘 다 dropout은 down_proj 출력 후 적용. activation은 `gpt.GPTConfig.mlpActivation`로 선택.
 */
class MLP(
    val embedDim: Int,
    useBias: Boolean = true,
    dropoutProbability: Float = 0.0f,
    private val activation: String = "gelu",
) {

    private val isSwiGLU: Boolean = activation.equals("swiglu", ignoreCase = true)

    /** SwiGLU 시 hidden = round(8/3 * embedDim) — Llama 표준. GELU 시 4*embedDim. */
    private val hiddenDim: Int =
        if (isSwiGLU) ((8 * embedDim + 1) / 3) else (4 * embedDim)

    // GELU 경로 레이어 (untouched 호환)
    val fullyConnected: Linear? =
        if (isSwiGLU) null else Linear(embedDim, hiddenDim, useBias)
    val projection: Linear? =
        if (isSwiGLU) null else Linear(hiddenDim, embedDim, useBias)

    // SwiGLU 경로 레이어
    val gateProjection: Linear? =
        if (isSwiGLU) Linear(embedDim, hiddenDim, useBias) else null
    val upProjection: Linear? =
        if (isSwiGLU) Linear(embedDim, hiddenDim, useBias) else null
    val downProjection: Linear? =
        if (isSwiGLU) Linear(hiddenDim, embedDim, useBias) else null

    val dropout: Dropout = Dropout(dropoutProbability)

    // GELU backward 캐시
    private var cachedH: Tensor? = null
    // SwiGLU backward 캐시
    private var cachedGate: Tensor? = null   // g = gate_proj(x)
    private var cachedUp: Tensor? = null     // u = up_proj(x)
    private var cachedSiluG: Tensor? = null  // silu(g)

    fun forward(x: Tensor): Tensor {
        return if (isSwiGLU) forwardSwiGLU(x) else forwardGELU(x)
    }

    fun backward(gy: Tensor): Tensor {
        return if (isSwiGLU) backwardSwiGLU(gy) else backwardGELU(gy)
    }

    private fun forwardGELU(x: Tensor): Tensor {
        val h = fullyConnected!!.forward(x)
        cachedH = h
        val a = gelu(h)
        val p = projection!!.forward(a)
        return dropout.forward(p)
    }

    private fun backwardGELU(gy: Tensor): Tensor {
        val h = cachedH ?: error("forward 없이 backward 호출")
        val dp = dropout.backward(gy)
        val da = projection!!.backward(dp)
        val dhData = geluBackward(h, da.data)
        val dh = Tensor(intArrayOf(da.rows, da.cols), dhData)
        return fullyConnected!!.backward(dh)
    }

    private fun forwardSwiGLU(x: Tensor): Tensor {
        val g = gateProjection!!.forward(x)
        val u = upProjection!!.forward(x)
        cachedGate = g
        cachedUp = u
        val sg = silu(g)
        cachedSiluG = sg
        // h = silu(g) ⊙ u
        val h = Tensor(g.shape.copyOf())
        for (i in 0 until g.numel) h.data[i] = sg.data[i] * u.data[i]
        val p = downProjection!!.forward(h)
        return dropout.forward(p)
    }

    private fun backwardSwiGLU(gy: Tensor): Tensor {
        val g = cachedGate ?: error("forward 없이 backward 호출")
        val u = cachedUp ?: error("forward 없이 backward 호출")
        val sg = cachedSiluG ?: error("forward 없이 backward 호출")

        val dp = dropout.backward(gy)                                // [N, C]
        val dh = downProjection!!.backward(dp)                       // [N, H]
        // h = sg ⊙ u  →  d_sg = dh * u,  d_u = dh * sg
        val dSgData = FloatArray(g.numel)
        val dUData = FloatArray(g.numel)
        for (i in 0 until g.numel) {
            dSgData[i] = dh.data[i] * u.data[i]
            dUData[i] = dh.data[i] * sg.data[i]
        }
        // d_g = silu'(g) * d_sg
        val dGData = siluBackward(g, dSgData)
        val dG = Tensor(g.shape.copyOf(), dGData)
        val dU = Tensor(u.shape.copyOf(), dUData)

        val dxFromGate = gateProjection!!.backward(dG)               // [N, C]
        val dxFromUp = upProjection!!.backward(dU)                   // [N, C]
        // 두 경로 합산
        val dx = Tensor(dxFromGate.shape.copyOf())
        for (i in 0 until dx.numel) dx.data[i] = dxFromGate.data[i] + dxFromUp.data[i]
        return dx
    }

    fun parameters(): List<Tensor> = if (isSwiGLU) {
        gateProjection!!.parameters() + upProjection!!.parameters() + downProjection!!.parameters()
    } else {
        fullyConnected!!.parameters() + projection!!.parameters()
    }
}
