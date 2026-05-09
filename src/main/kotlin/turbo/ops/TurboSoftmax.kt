package turbo.ops

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import turbo.TurboSimdMath
import turbo.TurboTensor
import kotlin.math.exp

/**
 * Row-wise softmax (max-shift + exp + normalize).
 *
 * Phase B: row reduction(max, sum) + scale을 SIMD化, exp는 `lanewise(EXP)`로 시도.
 * 미지원 플랫폼에선 자동 scalar fallback이라 정밀도는 표준 라이브러리와 동등.
 */
fun turboSoftmax(x: TurboTensor): TurboTensor {
    require(x.shape.size == 2) { "turboSoftmax는 2차원만: ${x.shape.contentToString()}" }
    val n = x.rows
    val c = x.cols
    val out = TurboTensor(intArrayOf(n, c))
    val xData = x.data
    val outData = out.data
    val species = TurboSimdMath.SPECIES
    val laneLen = species.length()
    val upper = species.loopBound(c)

    for (i in 0 until n) {
        val rowOff = i * c

        // 1) row max
        var maxAcc = FloatVector.broadcast(species, Float.NEGATIVE_INFINITY)
        var j = 0
        while (j < upper) {
            maxAcc = maxAcc.max(FloatVector.fromArray(species, xData, rowOff + j))
            j += laneLen
        }
        var maxVal = maxAcc.reduceLanes(VectorOperators.MAX)
        while (j < c) {
            val v = xData[rowOff + j]
            if (v > maxVal) maxVal = v
            j++
        }

        // 2) shifted exp + sum, store exp into out
        val vMax = FloatVector.broadcast(species, maxVal)
        var sumAcc = FloatVector.zero(species)
        j = 0
        while (j < upper) {
            val vX = FloatVector.fromArray(species, xData, rowOff + j)
            val vE = vX.sub(vMax).lanewise(VectorOperators.EXP)
            vE.intoArray(outData, rowOff + j)
            sumAcc = sumAcc.add(vE)
            j += laneLen
        }
        var sumExp = sumAcc.reduceLanes(VectorOperators.ADD)
        while (j < c) {
            val e = exp((xData[rowOff + j] - maxVal).toDouble()).toFloat()
            outData[rowOff + j] = e
            sumExp += e
            j++
        }

        // 3) normalize
        val inv = 1.0f / sumExp
        val vInv = FloatVector.broadcast(species, inv)
        j = 0
        while (j < upper) {
            FloatVector.fromArray(species, outData, rowOff + j).mul(vInv)
                .intoArray(outData, rowOff + j)
            j += laneLen
        }
        while (j < c) {
            outData[rowOff + j] *= inv
            j++
        }
    }
    return out
}

fun turboSoftmaxBackward(softmaxOut: TurboTensor, gyData: FloatArray): FloatArray {
    val n = softmaxOut.rows
    val c = softmaxOut.cols
    require(gyData.size == n * c)
    val dx = FloatArray(n * c)
    val sData = softmaxOut.data
    val species = TurboSimdMath.SPECIES
    val laneLen = species.length()
    val upper = species.loopBound(c)

    for (i in 0 until n) {
        val rowOff = i * c

        // dot = Σ s[j] * gy[j]
        var dotAcc = FloatVector.zero(species)
        var j = 0
        while (j < upper) {
            val vS = FloatVector.fromArray(species, sData, rowOff + j)
            val vGy = FloatVector.fromArray(species, gyData, rowOff + j)
            dotAcc = vS.fma(vGy, dotAcc)
            j += laneLen
        }
        var dot = dotAcc.reduceLanes(VectorOperators.ADD)
        while (j < c) {
            dot += sData[rowOff + j] * gyData[rowOff + j]
            j++
        }

        // dx[j] = s[j] * (gy[j] - dot)
        val vDot = FloatVector.broadcast(species, dot)
        j = 0
        while (j < upper) {
            val vS = FloatVector.fromArray(species, sData, rowOff + j)
            val vGy = FloatVector.fromArray(species, gyData, rowOff + j)
            vS.mul(vGy.sub(vDot)).intoArray(dx, rowOff + j)
            j += laneLen
        }
        while (j < c) {
            val s = sData[rowOff + j]
            dx[rowOff + j] = s * (gyData[rowOff + j] - dot)
            j++
        }
    }
    return dx
}
