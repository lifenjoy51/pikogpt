package vec.ops

import vec.Tensor
import kotlin.math.exp

/**
 * SiLU (Sigmoid-weighted Linear Unit) — SwiGLU MLP의 gate 활성화.
 *
 * 정의:    SiLU(x) = x · σ(x)        (σ = sigmoid)
 * 미분:    dSiLU/dx = σ(x) · (1 + x · (1 - σ(x)))
 *
 * GELU의 더 단순한 친척. Llama, PaLM 등 modern LM의 SwiGLU에서 사용.
 *
 * 원소별 연산. backward는 원본 x만 필요 (캐시 단순).
 */
fun silu(x: Tensor): Tensor {
    val out = Tensor(x.shape.copyOf())
    for (i in x.data.indices) {
        val v = x.data[i].toDouble()
        val s = 1.0 / (1.0 + exp(-v))
        out.data[i] = (v * s).toFloat()
    }
    return out
}

/** SiLU backward. `x`는 forward 입력, `gyData`는 출력 기울기, 반환은 입력 기울기. */
fun siluBackward(x: Tensor, gyData: FloatArray): FloatArray {
    require(gyData.size == x.numel)
    val dx = FloatArray(x.numel)
    for (i in x.data.indices) {
        val v = x.data[i].toDouble()
        val s = 1.0 / (1.0 + exp(-v))
        val localDerivative = s * (1.0 + v * (1.0 - s))
        dx[i] = (gyData[i].toDouble() * localDerivative).toFloat()
    }
    return dx
}
