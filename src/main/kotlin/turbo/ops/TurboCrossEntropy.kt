package turbo.ops

import kotlin.math.exp
import kotlin.math.ln
import turbo.TurboTensor

/**
 * Softmax + NLL fused cross-entropy. label smoothing + z-loss 옵션 지원.
 *
 * z-loss (PaLM/T5 표준 안정화):
 *   loss += zLossWeight * mean(lse²)  where lse = log(Σ exp(logits))
 *
 * z-loss는 logits의 매우 큰 값에 대한 페널티 — softmax 정규화 항(lse)이 폭주하지 않게 함.
 * default 0 → 기존 CE와 정확히 동일 (회귀 보장).
 */
data class TurboCrossEntropyResult(
    val loss: Float,
    val softmax: TurboTensor,
    /** z-loss backward에 필요한 행별 lse (zLossWeight=0이면 null). */
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
    var totalCeLoss = 0.0f
    var totalZLoss = 0.0f
    val epsOverV = if (labelSmoothing > 0f) labelSmoothing / v else 0f
    val oneMinusEps = 1.0f - labelSmoothing
    val lsePerRow = if (zLossWeight != 0f) FloatArray(n) else null

    for (i in 0 until n) {
        var maxVal = Float.NEGATIVE_INFINITY
        for (j in 0 until v) {
            val z = logits.data[i * v + j]
            if (z > maxVal) maxVal = z
        }
        var sumExp = 0.0f
        var sumLogits = 0.0f
        for (j in 0 until v) {
            val z = logits.data[i * v + j]
            val e = exp((z - maxVal).toDouble()).toFloat()
            sm.data[i * v + j] = e
            sumExp += e
            if (epsOverV > 0f) sumLogits += z
        }
        val invSum = 1.0f / sumExp
        for (j in 0 until v) sm.data[i * v + j] *= invSum

        val t = targets[i]
        require(t in 0 until v) { "target 범위 밖: $t vs vocab $v" }
        val lse = ln(sumExp.toDouble()).toFloat() + maxVal
        totalCeLoss += lse - oneMinusEps * logits.data[i * v + t] - epsOverV * sumLogits
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
    val factor = upstreamGrad / n
    val epsOverV = if (labelSmoothing > 0f) labelSmoothing / v else 0f
    val targetExtra = if (labelSmoothing > 0f) 1.0f - labelSmoothing else 1.0f
    val zActive = zLossWeight != 0f && lsePerRow != null
    for (i in 0 until n) {
        val zCoeff = if (zActive) 2.0f * zLossWeight * lsePerRow!![i] * factor else 0.0f
        for (j in 0 until v) {
            var g = (softmaxOut.data[i * v + j] - epsOverV) * factor
            if (zActive) g += zCoeff * softmaxOut.data[i * v + j]
            gLogits.data[i * v + j] = g
        }
        gLogits.data[i * v + targets[i]] -= targetExtra * factor
    }
    return gLogits
}
