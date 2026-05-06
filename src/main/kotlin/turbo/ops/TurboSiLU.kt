package turbo.ops

import turbo.TurboTensor
import kotlin.math.exp

/**
 * SiLU (x · σ(x)). SwiGLU의 gate 활성화. Phase 0은 vec와 동일.
 */
fun turboSilu(x: TurboTensor): TurboTensor {
    val out = TurboTensor(x.shape.copyOf())
    for (i in x.data.indices) {
        val v = x.data[i].toDouble()
        val s = 1.0 / (1.0 + exp(-v))
        out.data[i] = (v * s).toFloat()
    }
    return out
}

fun turboSiluBackward(x: TurboTensor, gyData: FloatArray): FloatArray {
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
