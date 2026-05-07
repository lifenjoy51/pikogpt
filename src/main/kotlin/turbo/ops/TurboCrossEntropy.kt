package turbo.ops

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import kotlin.math.exp
import kotlin.math.ln
import turbo.TurboSimdMath
import turbo.TurboTensor

/**
 * Softmax + NLL fused cross-entropy + label smoothing + z-loss. Phase B에서 SIMD化.
 *   loss = mean(lse - (1-ε)·logits[t] - (ε/V)·Σ logits) + zLossWeight*mean(lse²)
 */
data class TurboCrossEntropyResult(
    val loss: Float,
    val softmax: TurboTensor,
    val lsePerRow: FloatArray? = null,
) {
    override fun equals(other: Any?): Boolean = this === other
    override fun hashCode(): Int = System.identityHashCode(this)
}

fun turboCrossEntropyForward(
    logits: TurboTensor,
    targets: IntArray,
    labelSmoothing: Float = 0.0f,
    zLossWeight: Float = 0.0f,
): TurboCrossEntropyResult {
    require(logits.shape.size == 2)
    require(labelSmoothing in 0.0f..1.0f) { "labelSmoothing 범위 밖: $labelSmoothing" }
    val n = logits.rows
    val v = logits.cols
    require(targets.size == n) { "targets 크기가 N과 다름: ${targets.size} vs $n" }

    val sm = TurboTensor(intArrayOf(n, v))
    val smData = sm.data
    val logitsData = logits.data
    var totalCeLoss = 0.0f
    var totalZLoss = 0.0f
    val epsOverV = if (labelSmoothing > 0f) labelSmoothing / v else 0f
    val oneMinusEps = 1.0f - labelSmoothing
    val lsePerRow = if (zLossWeight != 0f) FloatArray(n) else null
    val species = TurboSimdMath.SPECIES
    val laneLen = species.length()
    val vUpper = species.loopBound(v)

    for (i in 0 until n) {
        val rowOff = i * v

        // 1) row max — SIMD reduceLanes(MAX)
        var maxAcc = FloatVector.broadcast(species, Float.NEGATIVE_INFINITY)
        var j = 0
        while (j < vUpper) {
            maxAcc = maxAcc.max(FloatVector.fromArray(species, logitsData, rowOff + j))
            j += laneLen
        }
        var maxVal = maxAcc.reduceLanes(VectorOperators.MAX)
        while (j < v) { if (logitsData[rowOff + j] > maxVal) maxVal = logitsData[rowOff + j]; j++ }

        // 2) sm[j] = exp(logits[j] - max), sumExp + sumLogits
        val vMax = FloatVector.broadcast(species, maxVal)
        // exp는 scalar fallback (lanewise EXP는 일부 플랫폼만)
        var sumExp = 0.0f
        var sumLogits = 0.0f
        j = 0
        while (j < v) {
            val z = logitsData[rowOff + j]
            val e = exp((z - maxVal).toDouble()).toFloat()
            smData[rowOff + j] = e
            sumExp += e
            if (epsOverV > 0f) sumLogits += z
            j++
        }

        // 3) sm *= 1/sumExp — SIMD scalar mul
        val invSum = 1.0f / sumExp
        val vInvSum = FloatVector.broadcast(species, invSum)
        j = 0
        while (j < vUpper) {
            FloatVector.fromArray(species, smData, rowOff + j).mul(vInvSum).intoArray(smData, rowOff + j)
            j += laneLen
        }
        while (j < v) { smData[rowOff + j] *= invSum; j++ }

        val t = targets[i]
        require(t in 0 until v) { "target 범위 밖: $t vs vocab $v" }
        val lse = ln(sumExp.toDouble()).toFloat() + maxVal
        totalCeLoss += lse - oneMinusEps * logitsData[rowOff + t] - epsOverV * sumLogits
        if (zLossWeight != 0f) {
            totalZLoss += zLossWeight * lse * lse
            lsePerRow!![i] = lse
        }
    }
    return TurboCrossEntropyResult(
        loss = (totalCeLoss + totalZLoss) / n,
        softmax = sm,
        lsePerRow = lsePerRow,
    )
}

fun turboCrossEntropyBackward(
    logits: TurboTensor,
    targets: IntArray,
    softmaxOut: TurboTensor,
    upstreamGrad: Float = 1.0f,
    labelSmoothing: Float = 0.0f,
    zLossWeight: Float = 0.0f,
    lsePerRow: FloatArray? = null,
): TurboTensor {
    val n = logits.rows
    val v = logits.cols
    val gLogits = TurboTensor(logits.shape.copyOf())
    val gData = gLogits.data
    val smData = softmaxOut.data
    val factor = upstreamGrad / n
    val epsOverV = if (labelSmoothing > 0f) labelSmoothing / v else 0f
    val targetExtra = if (labelSmoothing > 0f) 1.0f - labelSmoothing else 1.0f
    val zActive = zLossWeight != 0f && lsePerRow != null
    val species = TurboSimdMath.SPECIES
    val laneLen = species.length()
    val vUpper = species.loopBound(v)

    for (i in 0 until n) {
        val rowOff = i * v
        val zCoeff = if (zActive) 2.0f * zLossWeight * lsePerRow!![i] * factor else 0.0f
        // g[j] = sm[j]*(factor + zCoeff) - epsOverV*factor
        // (zActive=false면 g[j] = sm[j]*factor - epsOverV*factor)
        val combined = factor + zCoeff
        val vCombined = FloatVector.broadcast(species, combined)
        val vBias = FloatVector.broadcast(species, -epsOverV * factor)
        var j = 0
        while (j < vUpper) {
            val vSm = FloatVector.fromArray(species, smData, rowOff + j)
            vSm.fma(vCombined, vBias).intoArray(gData, rowOff + j)
            j += laneLen
        }
        while (j < v) {
            gData[rowOff + j] = smData[rowOff + j] * combined - epsOverV * factor
            j++
        }
        gData[rowOff + targets[i]] -= targetExtra * factor
    }
    return gLogits
}
