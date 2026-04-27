package vec.layer

import vec.Tensor
import kotlin.random.Random

/**
 * Inverted dropout 레이어.
 *
 *   Forward (training=true, p>0):
 *     mask[i] = 1 / (1 - p)  with probability (1 - p)
 *     mask[i] = 0            with probability p
 *     y = x ⊙ mask
 *
 *   Forward (training=false || p==0): identity (mask 생성 없음, 입력 그대로 반환)
 *
 *   Backward (training=true, p>0):
 *     dx = gy ⊙ mask  (forward에서 저장한 mask를 그대로 재사용)
 *
 *   Inverted 스킴인 덕분에 추론 경로는 scaling 없이 identity. 기대값 보존:
 *     E[y_i] = (1-p) * x_i * (1/(1-p)) + p * 0 = x_i.
 *
 * 파라미터 없음 (weight-less). 매 forward마다 새 mask를 뽑으므로 gradient accumulation이나
 * 병렬 worker들 사이에 mask를 공유하지 않는다 — 각 시퀀스가 독립적인 dropout 패턴을 본다.
 */
class VecDropout(val probability: Float) {
    /** true면 forward/backward에 mask 적용. false면 identity. Trainer/Sampler가 토글. */
    var training: Boolean = true

    /** forward에서 생성되어 backward에서 재사용되는 mask. shape은 입력과 동일. */
    private var cachedMask: FloatArray? = null
    private var cachedShape: IntArray? = null

    fun forward(x: Tensor): Tensor {
        if (!training || probability <= 0.0f) {
            // identity 경로: mask 생성/저장 없이 입력을 그대로 반환.
            cachedMask = null
            return x
        }
        val keepScale = 1.0f / (1.0f - probability)
        val mask = FloatArray(x.numel)
        for (i in mask.indices) {
            mask[i] = if (Random.Default.nextFloat() < probability) 0.0f else keepScale
        }
        cachedMask = mask
        cachedShape = x.shape

        val out = Tensor(x.shape.copyOf())
        for (i in out.data.indices) out.data[i] = x.data[i] * mask[i]
        return out
    }

    fun backward(gy: Tensor): Tensor {
        val mask = cachedMask
            ?: // identity 경로 — mask 없이 gy를 그대로 전달.
            return gy
        require(gy.numel == mask.size) { "backward gy shape != forward mask shape" }
        val out = Tensor(gy.shape.copyOf())
        for (i in out.data.indices) out.data[i] = gy.data[i] * mask[i]
        return out
    }

    fun parameters(): List<Tensor> = emptyList()
}
