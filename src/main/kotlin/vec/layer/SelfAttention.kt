package vec.layer

import vec.Tensor
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * 멀티-헤드 causal self-attention.
 *
 *   Notation:
 *     T = sequence length (토큰 수)
 *     C = embedding dim
 *     H = head 수
 *     D = head dim = C / H
 *
 *   Forward 개요:
 *     1) 입력 x[T, C]로부터 Q, K, V 각각 [T, C] 생성 (세 개의 Linear 투영)
 *     2) C 축을 [H, D]로 쪼개서 head별 Q_h, K_h, V_h [T, D]를 얻음
 *     3) 각 head에서
 *          scores_h = Q_h · K_h^T / √D        ← [T, T]
 *          causal mask:  j > i 위치를 -∞
 *          attn_h   = softmax(scores_h) row-wise
 *          out_h    = attn_h · V_h            ← [T, D]
 *     4) head를 다시 C축으로 합쳐 [T, C]
 *     5) y = out · Wo^T + bo
 *
 *   Backward: 위 단계들을 역순으로 chain rule. 각 head의 backward는
 *   softmax(Jacobian), matmul 두 번, scale 1/√D 로 이뤄진다.
 *
 * 구현 전략:
 *   - Q/K/V/O 투영은 `Linear`에 위임 — 파라미터/grad 관리를 공짜로 얻음.
 *   - softmax는 직접 내장 (op 호출보단 per-head loop가 읽기 쉬움).
 *   - causal mask는 softmax 전에 -∞를 쓰는 대신, exp 직전에 조건문으로 skip.
 *     (수치적으로 동등하면서 코드가 단순)
 */
class SelfAttention(
    val embedDim: Int,
    val numHeads: Int,
    useBias: Boolean = true,
) {
    init {
        require(embedDim % numHeads == 0) { "embedDim=$embedDim must be divisible by numHeads=$numHeads" }
    }

    val headDim: Int = embedDim / numHeads
    private val scale: Float = 1.0f / sqrt(headDim.toDouble()).toFloat()

    val qProjection: Linear = Linear(embedDim, embedDim, useBias)
    val kProjection: Linear = Linear(embedDim, embedDim, useBias)
    val vProjection: Linear = Linear(embedDim, embedDim, useBias)
    val outputProjection: Linear = Linear(embedDim, embedDim, useBias)

    // backward 재사용용 캐시 (한 번의 forward→backward 사이클 동안만 유효)
    private var cachedQ: Tensor? = null
    private var cachedK: Tensor? = null
    private var cachedV: Tensor? = null
    private var cachedAttnProbs: FloatArray? = null   // [H, T, T] row-major: h*T*T + i*T + j
    private var cachedT: Int = 0

    fun forward(x: Tensor): Tensor {
        val t = x.rows
        val c = x.cols
        require(c == embedDim)
        cachedT = t

        val q = qProjection.forward(x)
        val k = kProjection.forward(x)
        val v = vProjection.forward(x)
        cachedQ = q; cachedK = k; cachedV = v

        // out[T, C] 누적 버퍼
        val out = Tensor(intArrayOf(t, embedDim))
        val attnProbs = FloatArray(numHeads * t * t)

        for (h in 0 until numHeads) {
            val headOffset = h * headDim
            // 각 head에서 scores, softmax, out 계산
            // scores_h[i, j] = (Σ_d Q[i, headOffset+d] * K[j, headOffset+d]) * scale
            // causal: j > i 면 mask (softmax 입력에서 제외)
            for (i in 0 until t) {
                // 1) scores 행 i 계산 + max (수치 안정)
                val scoresRow = FloatArray(t)
                var maxScore = Float.NEGATIVE_INFINITY
                for (j in 0 until t) {
                    if (j > i) continue  // causal: 미래 토큰 무시
                    var dot = 0.0f
                    for (d in 0 until headDim) {
                        dot += q[i, headOffset + d] * k[j, headOffset + d]
                    }
                    val s = dot * scale
                    scoresRow[j] = s
                    if (s > maxScore) maxScore = s
                }
                // 2) exp(score - max), 합
                var sumExp = 0.0f
                for (j in 0..i) {
                    val e = exp((scoresRow[j] - maxScore).toDouble()).toFloat()
                    scoresRow[j] = e
                    sumExp += e
                }
                val invSum = 1.0f / sumExp
                for (j in 0..i) {
                    scoresRow[j] *= invSum
                    attnProbs[h * t * t + i * t + j] = scoresRow[j]
                }
                // j > i 위치는 0으로 유지 (이미 FloatArray 초기값 0)

                // 3) out_h 행 i = Σ_j attn[i, j] * V[j, head]
                for (d in 0 until headDim) {
                    var acc = 0.0f
                    for (j in 0..i) {
                        acc += scoresRow[j] * v[j, headOffset + d]
                    }
                    out[i, headOffset + d] = acc
                }
            }
        }

        cachedAttnProbs = attnProbs
        return outputProjection.forward(out)
    }

    fun backward(gy: Tensor): Tensor {
        val q = cachedQ ?: error("forward 전 backward")
        val k = cachedK!!
        val v = cachedV!!
        val attn = cachedAttnProbs!!
        val t = cachedT

        // 1) output projection backward → d_mergedOut [T, C]
        val dMerged = outputProjection.backward(gy)

        // 2) head별 backward
        val dQ = Tensor(intArrayOf(t, embedDim))
        val dK = Tensor(intArrayOf(t, embedDim))
        val dV = Tensor(intArrayOf(t, embedDim))

        for (h in 0 until numHeads) {
            val headOffset = h * headDim

            // 2a) d_out_h [T, D] ← dMerged의 h번째 슬라이스
            //     out_h = attn @ V_h  ⇒  d_attn[i, j] = Σ_d d_out_h[i, d] * V_h[j, d]
            //                           d_V_h[j, d]  += Σ_i attn[i, j] * d_out_h[i, d]
            val dAttn = FloatArray(t * t)  // 이 head의 [T, T] grad
            for (i in 0 until t) {
                for (j in 0..i) {
                    var acc = 0.0f
                    for (d in 0 until headDim) {
                        acc += dMerged[i, headOffset + d] * v[j, headOffset + d]
                    }
                    dAttn[i * t + j] = acc
                }
                for (d in 0 until headDim) {
                    // d_V_h[j, d] += Σ_i attn[i, j] * dMerged[i, headOffset+d]
                    // 루프 순서를 바꿔 j 바깥으로 두면 cache friendly
                }
            }
            // d_V 누적 (j, d 바깥, i 안)
            for (j in 0 until t) {
                for (d in 0 until headDim) {
                    var acc = 0.0f
                    for (i in j until t) {  // causal: j ≤ i 일 때만 attn이 0이 아님
                        acc += attn[h * t * t + i * t + j] * dMerged[i, headOffset + d]
                    }
                    dV[j, headOffset + d] += acc
                }
            }

            // 2b) softmax backward (행별 Jacobian)
            //     dScores[i, j] = attn[i, j] * ( dAttn[i, j] - Σ_k attn[i, k] * dAttn[i, k] )
            //     causal로 j > i 인 attn은 0이므로 그 부분 기여 없음.
            val dScores = FloatArray(t * t)
            for (i in 0 until t) {
                var dot = 0.0f
                for (j in 0..i) {
                    dot += attn[h * t * t + i * t + j] * dAttn[i * t + j]
                }
                for (j in 0..i) {
                    val s = attn[h * t * t + i * t + j]
                    dScores[i * t + j] = s * (dAttn[i * t + j] - dot)
                }
            }

            // 2c) scores = Q · K^T * scale
            //     dQ_h[i, d] += scale * Σ_j dScores[i, j] * K_h[j, d]
            //     dK_h[j, d] += scale * Σ_i dScores[i, j] * Q_h[i, d]
            for (i in 0 until t) {
                for (d in 0 until headDim) {
                    var acc = 0.0f
                    for (j in 0..i) {
                        acc += dScores[i * t + j] * k[j, headOffset + d]
                    }
                    dQ[i, headOffset + d] += scale * acc
                }
            }
            for (j in 0 until t) {
                for (d in 0 until headDim) {
                    var acc = 0.0f
                    for (i in j until t) {
                        acc += dScores[i * t + j] * q[i, headOffset + d]
                    }
                    dK[j, headOffset + d] += scale * acc
                }
            }
        }

        // 3) Q/K/V 투영 backward
        val dxQ = qProjection.backward(dQ)
        val dxK = kProjection.backward(dK)
        val dxV = vProjection.backward(dV)

        // 4) 세 경로의 grad 합산 → 입력 x의 grad
        val dx = Tensor(intArrayOf(t, embedDim))
        for (i in dx.data.indices) {
            dx.data[i] = dxQ.data[i] + dxK.data[i] + dxV.data[i]
        }
        return dx
    }

    fun parameters(): List<Tensor> =
        qProjection.parameters() +
                kProjection.parameters() +
                vProjection.parameters() +
                outputProjection.parameters()
}
