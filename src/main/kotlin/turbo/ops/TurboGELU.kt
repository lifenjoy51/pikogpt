package turbo.ops

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import turbo.TurboSimdMath
import turbo.TurboTensor
import kotlin.math.tanh

/**
 * GELU tanh 근사.
 *
 *   GELU(x) ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
 *
 * Phase B: element-wise SIMD. tanh는 `lanewise(TANH)`로 시도 — 미지원 플랫폼에선 scalar fallback.
 */
private const val GELU_A: Float = 0.7978845608028654f
private const val GELU_K: Float = 0.044715f

fun turboGelu(x: TurboTensor): TurboTensor {
    val out = TurboTensor(x.shape.copyOf())
    val len = x.numel
    val xData = x.data
    val outData = out.data
    val species = TurboSimdMath.SPECIES
    val laneLen = species.length()
    val upper = species.loopBound(len)
    val vA = FloatVector.broadcast(species, GELU_A)
    val vK = FloatVector.broadcast(species, GELU_K)
    val vHalf = FloatVector.broadcast(species, 0.5f)
    val vOne = FloatVector.broadcast(species, 1.0f)

    var i = 0
    while (i < upper) {
        val vX = FloatVector.fromArray(species, xData, i)
        val vX2 = vX.mul(vX)
        val vX3 = vX2.mul(vX)
        // inner = A * (x + K * x³)
        val vInner = vA.mul(vX.add(vK.mul(vX3)))
        val vTanh = vInner.lanewise(VectorOperators.TANH)
        // y = 0.5 * x * (1 + tanh(inner))
        vHalf.mul(vX).mul(vOne.add(vTanh)).intoArray(outData, i)
        i += laneLen
    }
    while (i < len) {
        val v = xData[i]
        val inner = GELU_A * (v + GELU_K * v * v * v)
        outData[i] = 0.5f * v * (1.0f + tanh(inner.toDouble()).toFloat())
        i++
    }
    return out
}

fun turboGeluBackward(x: TurboTensor, gyData: FloatArray): FloatArray {
    require(gyData.size == x.numel)
    val len = x.numel
    val xData = x.data
    val dx = FloatArray(len)
    val species = TurboSimdMath.SPECIES
    val laneLen = species.length()
    val upper = species.loopBound(len)
    val vA = FloatVector.broadcast(species, GELU_A)
    val vK = FloatVector.broadcast(species, GELU_K)
    val vK3 = FloatVector.broadcast(species, 3.0f * GELU_K)
    val vHalf = FloatVector.broadcast(species, 0.5f)
    val vOne = FloatVector.broadcast(species, 1.0f)

    var i = 0
    while (i < upper) {
        val vX = FloatVector.fromArray(species, xData, i)
        val vGy = FloatVector.fromArray(species, gyData, i)
        val vX2 = vX.mul(vX)
        val vX3 = vX2.mul(vX)
        val vInner = vA.mul(vX.add(vK.mul(vX3)))
        val vTanh = vInner.lanewise(VectorOperators.TANH)
        // innerPrime = A * (1 + 3K * x²)
        val vInnerPrime = vA.mul(vOne.add(vK3.mul(vX2)))
        // localDerivative = 0.5 * (1 + tanh) + 0.5 * x * (1 - tanh²) * innerPrime
        val vTermA = vHalf.mul(vOne.add(vTanh))
        val vTermB = vHalf.mul(vX).mul(vOne.sub(vTanh.mul(vTanh))).mul(vInnerPrime)
        val vDeriv = vTermA.add(vTermB)
        vGy.mul(vDeriv).intoArray(dx, i)
        i += laneLen
    }
    while (i < len) {
        val v = xData[i]
        val inner = GELU_A * (v + GELU_K * v * v * v)
        val t = tanh(inner.toDouble()).toFloat()
        val innerPrime = GELU_A * (1.0f + 3.0f * GELU_K * v * v)
        val localDerivative = 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * innerPrime
        dx[i] = gyData[i] * localDerivative
        i++
    }
    return dx
}
