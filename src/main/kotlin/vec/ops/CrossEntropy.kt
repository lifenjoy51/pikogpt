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

fun crossEntropyForward(logits: Tensor, targets: IntArray): CrossEntropyResult {
    require(logits.shape.size == 2)
    val n = logits.rows
    val v = logits.cols
    require(targets.size == n) { "targets 크기가 N과 다름: ${targets.size} vs $n" }

    val sm = Tensor(intArrayOf(n, v))
    var totalLoss = 0.0f

    for (i in 0 until n) {
        // 행별 max
        var maxVal = Float.NEGATIVE_INFINITY
        for (j in 0 until v) {
            val z = logits.data[i * v + j]
            if (z > maxVal) maxVal = z
        }
        // exp(z - max), 합
        var sumExp = 0.0f
        for (j in 0 until v) {
            val e = exp((logits.data[i * v + j] - maxVal).toDouble()).toFloat()
            sm.data[i * v + j] = e
            sumExp += e
        }
        val invSum = 1.0f / sumExp
        for (j in 0 until v) sm.data[i * v + j] *= invSum

        // loss: -log(softmax[target]) = log(sumExp) + max - logit[target]
        val t = targets[i]
        require(t in 0 until v) { "target 범위 밖: $t vs vocab $v" }
        totalLoss += ln(sumExp.toDouble()).toFloat() + maxVal - logits.data[i * v + t]
    }
    return CrossEntropyResult(loss = totalLoss / n, softmax = sm)
}

/**
 * Cross-entropy backward. `logits`에 대한 기울기를 **새 Tensor로 반환**한다.
 *
 *   dLogits[i, j] = upstreamGrad * ( softmax[i, j] - 1_{j==target_i} ) / N
 *
 * - `upstreamGrad`: 상위 그래프에서 내려오는 loss의 스칼라 기울기. 단일 loss를 backward할 때는 1.
 *   gradient accumulation / batch 평균 등을 처리하려면 외부에서 1/(A*B) 같은 스케일을 넘겨주면 된다.
 * - 반환 Tensor의 shape는 logits와 동일. 호출자가 이 Tensor를 그대로 model.backward()에 전달.
 */
fun crossEntropyBackward(
    logits: Tensor,
    targets: IntArray,
    softmaxOut: Tensor,
    upstreamGrad: Float = 1.0f,
): Tensor {
    val n = logits.rows
    val v = logits.cols
    val gLogits = Tensor(logits.shape.copyOf())
    val factor = upstreamGrad / n
    for (i in 0 until n) {
        for (j in 0 until v) {
            gLogits.data[i * v + j] = softmaxOut.data[i * v + j] * factor
        }
        gLogits.data[i * v + targets[i]] -= factor
    }
    return gLogits
}
