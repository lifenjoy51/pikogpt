package turbo.layer

import turbo.TurboTensor
import turbo.ops.turboGelu
import turbo.ops.turboGeluBackward
import turbo.ops.turboSilu
import turbo.ops.turboSiluBackward

/**
 * Transformer FFN. Phase 0은 vec와 동일 (GELU 또는 SwiGLU 분기).
 *   GELU:    fc(C→4C) → gelu → proj(4C→C) → dropout
 *   SwiGLU:  silu(gate_proj(x)) ⊙ up_proj(x) → down_proj → dropout, hidden = ⌊8C/3⌋
 */
class TurboMLP(
    val embedDim: Int,
    useBias: Boolean = true,
    dropoutProbability: Float = 0.0f,
    private val activation: String = "gelu",
) {
    private val isSwiGLU: Boolean = activation.equals("swiglu", ignoreCase = true)

    private val hiddenDim: Int =
        if (isSwiGLU) ((8 * embedDim + 1) / 3) else (4 * embedDim)

    val fullyConnected: TurboLinear? =
        if (isSwiGLU) null else TurboLinear(embedDim, hiddenDim, useBias)
    val projection: TurboLinear? =
        if (isSwiGLU) null else TurboLinear(hiddenDim, embedDim, useBias)

    val gateProjection: TurboLinear? =
        if (isSwiGLU) TurboLinear(embedDim, hiddenDim, useBias) else null
    val upProjection: TurboLinear? =
        if (isSwiGLU) TurboLinear(embedDim, hiddenDim, useBias) else null
    val downProjection: TurboLinear? =
        if (isSwiGLU) TurboLinear(hiddenDim, embedDim, useBias) else null

    val dropout: TurboDropout = TurboDropout(dropoutProbability)

    private var cachedH: TurboTensor? = null
    private var cachedGate: TurboTensor? = null
    private var cachedUp: TurboTensor? = null
    private var cachedSiluG: TurboTensor? = null

    fun forward(x: TurboTensor): TurboTensor =
        if (isSwiGLU) forwardSwiGLU(x) else forwardGELU(x)

    fun backward(gy: TurboTensor): TurboTensor =
        if (isSwiGLU) backwardSwiGLU(gy) else backwardGELU(gy)

    private fun forwardGELU(x: TurboTensor): TurboTensor {
        val h = fullyConnected!!.forward(x)
        cachedH = h
        val a = turboGelu(h)
        val p = projection!!.forward(a)
        return dropout.forward(p)
    }

    private fun backwardGELU(gy: TurboTensor): TurboTensor {
        val h = cachedH ?: error("forward 없이 backward 호출")
        val dp = dropout.backward(gy)
        val da = projection!!.backward(dp)
        val dhData = turboGeluBackward(h, da.data)
        val dh = TurboTensor(intArrayOf(da.rows, da.cols), dhData)
        return fullyConnected!!.backward(dh)
    }

    private fun forwardSwiGLU(x: TurboTensor): TurboTensor {
        val g = gateProjection!!.forward(x)
        val u = upProjection!!.forward(x)
        cachedGate = g
        cachedUp = u
        val sg = turboSilu(g)
        cachedSiluG = sg
        val h = TurboTensor(g.shape.copyOf())
        for (i in 0 until g.numel) h.data[i] = sg.data[i] * u.data[i]
        val p = downProjection!!.forward(h)
        return dropout.forward(p)
    }

    private fun backwardSwiGLU(gy: TurboTensor): TurboTensor {
        val g = cachedGate ?: error("forward 없이 backward 호출")
        val u = cachedUp ?: error("forward 없이 backward 호출")
        val sg = cachedSiluG ?: error("forward 없이 backward 호출")

        val dp = dropout.backward(gy)
        val dh = downProjection!!.backward(dp)
        val dSgData = FloatArray(g.numel)
        val dUData = FloatArray(g.numel)
        for (i in 0 until g.numel) {
            dSgData[i] = dh.data[i] * u.data[i]
            dUData[i] = dh.data[i] * sg.data[i]
        }
        val dGData = turboSiluBackward(g, dSgData)
        val dG = TurboTensor(g.shape.copyOf(), dGData)
        val dU = TurboTensor(u.shape.copyOf(), dUData)

        val dxFromGate = gateProjection!!.backward(dG)
        val dxFromUp = upProjection!!.backward(dU)
        val dx = TurboTensor(dxFromGate.shape.copyOf())
        for (i in 0 until dx.numel) dx.data[i] = dxFromGate.data[i] + dxFromUp.data[i]
        return dx
    }

    fun parameters(): List<TurboTensor> = if (isSwiGLU) {
        gateProjection!!.parameters() + upProjection!!.parameters() + downProjection!!.parameters()
    } else {
        fullyConnected!!.parameters() + projection!!.parameters()
    }
}
