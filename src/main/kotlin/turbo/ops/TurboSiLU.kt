package turbo.ops

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import turbo.TurboSimdMath
import turbo.TurboTensor
import kotlin.math.exp

/**
 * SiLU (x · σ(x)). SwiGLU의 gate 활성화.
 *
 * Phase B: element-wise SIMD. exp는 `lanewise(EXP)`로 시도 — hardware SIMD 미지원 플랫폼에선
 * 자동 scalar fallback (Apple Silicon NEON 포함). 정밀도는 표준 라이브러리와 동등.
 */
fun turboSilu(x: TurboTensor): TurboTensor {
    val out = TurboTensor(x.shape.copyOf())
    val len = x.numel
    val xData = x.data
    val outData = out.data
    val species = TurboSimdMath.SPECIES
    val laneLen = species.length()
    val upper = species.loopBound(len)
    val vOne = FloatVector.broadcast(species, 1.0f)

    var i = 0
    while (i < upper) {
        val vX = FloatVector.fromArray(species, xData, i)
        val vExp = vX.neg().lanewise(VectorOperators.EXP)
        val vS = vOne.div(vOne.add(vExp))
        vX.mul(vS).intoArray(outData, i)
        i += laneLen
    }
    while (i < len) {
        val v = xData[i].toDouble()
        val s = 1.0 / (1.0 + exp(-v))
        outData[i] = (v * s).toFloat()
        i++
    }
    return out
}

fun turboSiluBackward(x: TurboTensor, gyData: FloatArray): FloatArray {
    require(gyData.size == x.numel)
    val len = x.numel
    val xData = x.data
    val dx = FloatArray(len)
    val species = TurboSimdMath.SPECIES
    val laneLen = species.length()
    val upper = species.loopBound(len)
    val vOne = FloatVector.broadcast(species, 1.0f)

    var i = 0
    while (i < upper) {
        val vX = FloatVector.fromArray(species, xData, i)
        val vGy = FloatVector.fromArray(species, gyData, i)
        val vExp = vX.neg().lanewise(VectorOperators.EXP)
        val vS = vOne.div(vOne.add(vExp))
        // localDerivative = s * (1 + x * (1 - s))
        val vDeriv = vS.mul(vOne.add(vX.mul(vOne.sub(vS))))
        vGy.mul(vDeriv).intoArray(dx, i)
        i += laneLen
    }
    while (i < len) {
        val v = xData[i].toDouble()
        val s = 1.0 / (1.0 + exp(-v))
        val localDerivative = s * (1.0 + v * (1.0 - s))
        dx[i] = (gyData[i].toDouble() * localDerivative).toFloat()
    }
    return dx
}
