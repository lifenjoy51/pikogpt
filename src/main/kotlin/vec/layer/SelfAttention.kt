package vec.layer

import vec.Tensor
import vec.ops.applyRoPE
import vec.ops.applyRoPEBackward
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
    dropoutProbability: Float = 0.0f,
    /** "learned"(기본) | "rope" — RoPE 시 Q와 K projection 직후 위치별 회전 적용. */
    private val positionEncoding: String = "learned",
) {
    init {
        require(embedDim % numHeads == 0) { "embedDim=$embedDim must be divisible by numHeads=$numHeads" }
    }

    private val useRoPE: Boolean = positionEncoding.equals("rope", ignoreCase = true)

    val headDim: Int = embedDim / numHeads
    private val scale: Float = 1.0f / sqrt(headDim.toDouble()).toFloat()

    val qProjection: Linear = Linear(embedDim, embedDim, useBias)
    val kProjection: Linear = Linear(embedDim, embedDim, useBias)
    val vProjection: Linear = Linear(embedDim, embedDim, useBias)
    val outputProjection: Linear = Linear(embedDim, embedDim, useBias)

    /**
     * Attention dropout — softmax(scores)로 얻은 attention weights에 적용.
     * residual dropout — output projection 뒤 최종 출력에 적용 (residual 합산 전).
     * 표준 GPT-2 convention.
     */
    val attnDropout: Dropout = Dropout(dropoutProbability)
    val residDropout: Dropout = Dropout(dropoutProbability)

    // backward 재사용용 캐시 (한 번의 forward→backward 사이클 동안만 유효)
    private var cachedQ: Tensor? = null
    private var cachedK: Tensor? = null
    private var cachedV: Tensor? = null
    private var cachedAttnProbs: FloatArray? = null        // softmax 출력 (pre-dropout), softmax backward용
    private var cachedDroppedAttn: FloatArray? = null      // post-dropout attn, d_V / d_attn_dropped 계산용
    private var cachedT: Int = 0

    fun forward(x: Tensor): Tensor {
        val t = x.rows
        val c = x.cols
        require(c == embedDim)
        cachedT = t

        val q = qProjection.forward(x)
        val k = kProjection.forward(x)
        val v = vProjection.forward(x)
        // RoPE: Q와 K에 위치별 회전 적용 (V는 회전 안 함). in-place 수정.
        if (useRoPE) {
            applyRoPE(q, numHeads)
            applyRoPE(k, numHeads)
        }
        cachedQ = q; cachedK = k; cachedV = v

        // 1) softmax(scores)까지만 먼저 계산 — attention dropout이 attn_probs에 들어가야 하므로
        //    out = attn · V 누적은 dropout 이후에 별도 루프로 수행.
        val attnProbs = FloatArray(numHeads * t * t)
        for (h in 0 until numHeads) {
            val headOffset = h * headDim
            for (i in 0 until t) {
                // scores_h[i, j] = (Σ_d Q[i, d] * K[j, d]) * scale, j > i는 causal mask로 제외
                val scoresRow = FloatArray(t)
                var maxScore = Float.NEGATIVE_INFINITY
                for (j in 0..i) {
                    var dot = 0.0f
                    for (d in 0 until headDim) {
                        dot += q[i, headOffset + d] * k[j, headOffset + d]
                    }
                    val s = dot * scale
                    scoresRow[j] = s
                    if (s > maxScore) maxScore = s
                }
                var sumExp = 0.0f
                for (j in 0..i) {
                    val e = exp((scoresRow[j] - maxScore).toDouble()).toFloat()
                    scoresRow[j] = e
                    sumExp += e
                }
                val invSum = 1.0f / sumExp
                for (j in 0..i) {
                    attnProbs[h * t * t + i * t + j] = scoresRow[j] * invSum
                }
                // j > i 위치는 0 (FloatArray 초기값 유지)
            }
        }
        // backward에서 softmax 역전파에 쓸 pre-dropout attn 보존.
        cachedAttnProbs = attnProbs

        // 2) Attention dropout — attn_probs 전체를 [H*T*T] 평탄 Tensor로 감싸 Dropout에 위임.
        //    inverted dropout이라 training=false이면 identity, p=0이면 identity.
        val attnTensor = Tensor(intArrayOf(numHeads * t * t), attnProbs)
        val droppedAttnTensor = attnDropout.forward(attnTensor)
        val droppedAttn = droppedAttnTensor.data
        cachedDroppedAttn = droppedAttn

        // 3) out_h = attn_dropped · V_h
        val out = Tensor(intArrayOf(t, embedDim))
        for (h in 0 until numHeads) {
            val headOffset = h * headDim
            for (i in 0 until t) {
                for (d in 0 until headDim) {
                    var acc = 0.0f
                    for (j in 0..i) {
                        acc += droppedAttn[h * t * t + i * t + j] * v[j, headOffset + d]
                    }
                    out[i, headOffset + d] = acc
                }
            }
        }

        // 4) Output projection + residual dropout.
        val projected = outputProjection.forward(out)
        return residDropout.forward(projected)
    }

    fun backward(gy: Tensor): Tensor {
        val q = cachedQ ?: error("forward 전 backward")
        val k = cachedK!!
        val v = cachedV!!
        val attn = cachedAttnProbs!!            // pre-dropout softmax 출력 (softmax backward 기준)
        val droppedAttn = cachedDroppedAttn!!   // post-dropout attn (d_attn_dropped 계산의 기준은 droppedAttn)
        val t = cachedT

        // 1) residual dropout backward → output projection backward → d_mergedOut [T, C]
        val dResid = residDropout.backward(gy)
        val dMerged = outputProjection.backward(dResid)

        // 2) head별 backward — out = attn_dropped · V 이므로
        //    d_attn_dropped[i, j] = Σ_d dMerged[i, d] * V[j, d]
        //    d_V[j, d]           += Σ_i attn_dropped[i, j] * dMerged[i, d]
        val dQ = Tensor(intArrayOf(t, embedDim))
        val dK = Tensor(intArrayOf(t, embedDim))
        val dV = Tensor(intArrayOf(t, embedDim))

        // attn dropout의 mask를 재사용하려면 평탄 Tensor를 거쳐야 하므로
        // 먼저 모든 head의 d_attn_dropped를 [H*T*T]로 모았다가 한 번에 dropout backward.
        val dAttnDropped = FloatArray(numHeads * t * t)

        for (h in 0 until numHeads) {
            val headOffset = h * headDim

            // 2a) d_attn_dropped: out = attn_dropped @ V
            for (i in 0 until t) {
                for (j in 0..i) {
                    var acc = 0.0f
                    for (d in 0 until headDim) {
                        acc += dMerged[i, headOffset + d] * v[j, headOffset + d]
                    }
                    dAttnDropped[h * t * t + i * t + j] = acc
                }
            }
            // 2b) d_V — attn_dropped (post-dropout)를 그대로 가중치로 사용.
            for (j in 0 until t) {
                for (d in 0 until headDim) {
                    var acc = 0.0f
                    for (i in j until t) {  // causal: j ≤ i 일 때만 attn이 0이 아님
                        acc += droppedAttn[h * t * t + i * t + j] * dMerged[i, headOffset + d]
                    }
                    dV[j, headOffset + d] += acc
                }
            }
        }

        // 2c) Attention dropout backward (d_attn_dropped → d_attn). mask 재사용.
        val dAttnAll = attnDropout.backward(Tensor(intArrayOf(numHeads * t * t), dAttnDropped)).data

        for (h in 0 until numHeads) {
            val headOffset = h * headDim
            val dAttn = FloatArray(t * t)
            for (i in 0 until t) for (j in 0..i) {
                dAttn[i * t + j] = dAttnAll[h * t * t + i * t + j]
            }

            // 2d) softmax backward (행별 Jacobian) — attn은 pre-dropout 확률.
            //     dScores[i, j] = attn[i, j] * ( dAttn[i, j] - Σ_k attn[i, k] * dAttn[i, k] )
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

            // 2e) scores = Q · K^T * scale
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

        // RoPE backward — Q와 K에 in-place 역회전 적용 (V는 무관).
        if (useRoPE) {
            applyRoPEBackward(dQ, numHeads)
            applyRoPEBackward(dK, numHeads)
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
