package turbo.ops

import turbo.TurboTensor
import kotlin.math.tanh

/**
 * GELU tanh 근사. Phase 0은 vec와 동일.
 *
 *   GELU(x) ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
 */
private const val GELU_A: Float = 0.7978845608028654f
private const val GELU_K: Float = 0.044715f

fun turboGelu(x: TurboTensor): TurboTensor {
    val out = TurboTensor(x.shape.copyOf())
    for (i in x.data.indices) {
        val v = x.data[i]
        val inner = GELU_A * (v + GELU_K * v * v * v)
        out.data[i] = 0.5f * v * (1.0f + tanh(inner.toDouble()).toFloat())
    }
    return out
}

fun turboGeluBackward(x: TurboTensor, gyData: FloatArray): FloatArray {
    require(gyData.size == x.numel)
    val dx = FloatArray(x.numel)
    for (i in x.data.indices) {
        val v = x.data[i]
        val inner = GELU_A * (v + GELU_K * v * v * v)
        val t = tanh(inner.toDouble()).toFloat()
        val innerPrime = GELU_A * (1.0f + 3.0f * GELU_K * v * v)
        val localDerivative = 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * innerPrime
        dx[i] = gyData[i] * localDerivative
    }
    return dx
}
