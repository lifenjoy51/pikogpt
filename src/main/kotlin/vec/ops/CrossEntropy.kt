package vec.ops

import kotlin.math.exp
import kotlin.math.ln
import vec.Tensor

/**
 * Softmax와 Negative Log-Likelihood를 합친 수치 안정한 cross-entropy loss.
 *
 * 입력:
 *   - `logits`  shape [N, V] — 각 위치 i의 vocab V에 대한 미정규 점수
 *   - `targets` IntArray(N) — 각 위치의 정답 토큰 id
 *
 * Forward:
 *   per_i_loss = log( Σ_j exp(logits[i, j] - m_i) ) + m_i - logits[i, target_i]
 *             (m_i = max_j logits[i, j], 오버플로 방지)
 *   loss = mean over N
 *
 * Backward (softmax + NLL의 유명한 결합 형태):
 *   ∂loss/∂logits[i, j] = ( softmax(logits)[i, j] - onehot(target_i)[j] ) / N
 *
 * softmax를 별도로 호출하지 않고 한 번에 처리하기 때문에 수치적으로도 가장 안정적이고
 * 가장 효율적인 경로이다.
 */
data class CrossEntropyResult(
    val loss: Float,
    /** shape [N, V]의 softmax 출력. backward 캐시로 쓰인다. */
    val softmax: Tensor,
)

fun crossEntropyForward(
    logits: Tensor,
    targets: IntArray,
    labelSmoothing: Float = 0.0f,
): CrossEntropyResult {
    require(logits.shape.size == 2)
    require(labelSmoothing in 0.0f..1.0f) { "labelSmoothing 범위 밖: $labelSmoothing" }
    val n = logits.rows
    val v = logits.cols
    require(targets.size == n) { "targets 크기가 N과 다름: ${targets.size} vs $n" }

    val sm = Tensor(intArrayOf(n, v))
    var totalLoss = 0.0f

    // Label smoothing ε>0이면 target 분포가 (1-ε)·onehot + ε·uniform.
    //   q[target] = 1 - ε + ε/V,  q[j≠target] = ε/V
    //   loss per i = -Σ_j q[j] * log(softmax[j])
    //             = (1-ε)·(-log sm[target]) + ε · H(uniform, sm)
    //             = (1-ε)·NLL_target + ε·(log_sum_exp - mean_logit_no_max_shift)
    // 수치 안정 위해 log-sum-exp 공식으로 전개:
    //   log sm[j] = (logits[j] - max) - log(sumExp)
    //   -Σ q log sm = -(1-ε)*(logits[t]-max-lse) - (ε/V)*Σ_j(logits[j]-max-lse)
    //   = -(1-ε)*(logits[t]-max)+(1-ε)*lse - (ε/V)*(Σ_j logits[j] - V*max) + ε*lse
    //   = lse + max - (1-ε)*logits[t] - (ε/V)*Σ_j logits[j]                  (max 상쇄)
    // ε=0이면 원래 식과 동일.
    val epsOverV = if (labelSmoothing > 0f) labelSmoothing / v else 0f
    val oneMinusEps = 1.0f - labelSmoothing

    for (i in 0 until n) {
        // 행별 max
        var maxVal = Float.NEGATIVE_INFINITY
        for (j in 0 until v) {
            val z = logits.data[i * v + j]
            if (z > maxVal) maxVal = z
        }
        // exp(z - max), 합
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
        totalLoss += lse - oneMinusEps * logits.data[i * v + t] - epsOverV * sumLogits
    }
    return CrossEntropyResult(loss = totalLoss / n, softmax = sm)
}

/**
 * Cross-entropy backward. `logits`에 대한 기울기를 **새 Tensor로 반환**한다.
 *
 * Smoothing ε로 일반화한 형태:
 *   q[target] = 1 - ε + ε/V,  q[j≠target] = ε/V
 *   dLogits[i, j] = upstreamGrad * ( softmax[i, j] - q[j] ) / N
 *   즉 ε=0이면 원래 공식 (softmax - onehot)/N.
 */
fun crossEntropyBackward(
    logits: Tensor,
    targets: IntArray,
    softmaxOut: Tensor,
    upstreamGrad: Float = 1.0f,
    labelSmoothing: Float = 0.0f,
): Tensor {
    val n = logits.rows
    val v = logits.cols
    val gLogits = Tensor(logits.shape.copyOf())
    val factor = upstreamGrad / n
    val epsOverV = if (labelSmoothing > 0f) labelSmoothing / v else 0f
    val targetExtra = if (labelSmoothing > 0f) 1.0f - labelSmoothing else 1.0f
    for (i in 0 until n) {
        for (j in 0 until v) {
            // grad = softmax - q[j]
            gLogits.data[i * v + j] = (softmaxOut.data[i * v + j] - epsOverV) * factor
        }
        gLogits.data[i * v + targets[i]] -= targetExtra * factor
    }
    return gLogits
}
