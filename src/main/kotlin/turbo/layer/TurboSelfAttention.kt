package turbo.layer

import turbo.TurboKVCache
import turbo.TurboTensor
import turbo.ops.turboApplyRoPE
import turbo.ops.turboApplyRoPEAtPosition
import turbo.ops.turboApplyRoPEBackward
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * 멀티-헤드 causal self-attention.
 *
 * Phase 0은 vec와 동일 (naive Q·K^T → softmax → ·V).
 * Phase 1에서 옵션 인자(numKvHeads, useFusedQkv, useQkNorm)를 받지만 실제 구현은 단계적으로
 * 추가된다. Default OFF (numKvHeads=numHeads, useFusedQkv=false, useQkNorm=false)면 Phase 0
 * 경로 그대로 동작.
 *
 * Phase 3에서 Flash Attention v2로 교체.
 */
class TurboSelfAttention(
    val embedDim: Int,
    val numHeads: Int,
    useBias: Boolean = true,
    dropoutProbability: Float = 0.0f,
    private val positionEncoding: String = "learned",
    val numKvHeads: Int = numHeads,
    val useFusedQkv: Boolean = false,
    val useQkNorm: Boolean = false,
    private val normalizationType: String = "layernorm",
) {
    init {
        require(embedDim % numHeads == 0) { "embedDim=$embedDim must be divisible by numHeads=$numHeads" }
        require(numHeads % numKvHeads == 0) { "numHeads=$numHeads must be divisible by numKvHeads=$numKvHeads" }
        require(numKvHeads in 1..numHeads) { "numKvHeads=$numKvHeads must be in [1, numHeads=$numHeads]" }
    }

    private val useRoPE: Boolean = positionEncoding.equals("rope", ignoreCase = true)

    val headDim: Int = embedDim / numHeads
    /** GQA 시 K/V dimension. MHA(numKvHeads=numHeads)면 embedDim과 동일. */
    val kvDim: Int = numKvHeads * headDim
    /** GQA group size — 같은 KV head를 공유하는 Q head의 수. */
    private val groupSize: Int = numHeads / numKvHeads
    private val scale: Float = 1.0f / sqrt(headDim.toDouble()).toFloat()

    /** useFusedQkv=false 경로 — 3개 별도 projection. K/V는 GQA 시 kvDim 출력. */
    val qProjection: TurboLinear? = if (!useFusedQkv) TurboLinear(embedDim, embedDim, useBias) else null
    val kProjection: TurboLinear? = if (!useFusedQkv) TurboLinear(embedDim, kvDim, useBias) else null
    val vProjection: TurboLinear? = if (!useFusedQkv) TurboLinear(embedDim, kvDim, useBias) else null

    /** useFusedQkv=true 경로 — 단일 projection (out=embedDim+2*kvDim). */
    val qkvProjection: TurboLinear? = if (useFusedQkv) TurboLinear(embedDim, embedDim + 2 * kvDim, useBias) else null

    val outputProjection: TurboLinear = TurboLinear(embedDim, embedDim, useBias)

    val attnDropout: TurboDropout = TurboDropout(dropoutProbability)
    val residDropout: TurboDropout = TurboDropout(dropoutProbability)

    /** qk-norm — Q/K projection 직후 head_dim 단위로 RMSNorm 적용 (RoPE 전). null이면 skip. */
    val qNorm: TurboNorm? = if (useQkNorm) createTurboNorm(headDim, useBias = false, normalizationType) else null
    val kNorm: TurboNorm? = if (useQkNorm) createTurboNorm(headDim, useBias = false, normalizationType) else null

    private var cachedQ: TurboTensor? = null
    private var cachedK: TurboTensor? = null
    private var cachedV: TurboTensor? = null
    private var cachedAttnProbs: FloatArray? = null
    private var cachedDroppedAttn: FloatArray? = null
    private var cachedT: Int = 0

    fun forward(x: TurboTensor): TurboTensor {
        val t = x.rows
        val c = x.cols
        require(c == embedDim)
        cachedT = t

        var q: TurboTensor
        var k: TurboTensor
        val v: TurboTensor
        if (useFusedQkv) {
            val qkv = qkvProjection!!.forward(x)  // [T, embedDim + 2*kvDim]
            val triple = sliceQkv(qkv, embedDim, kvDim)
            q = triple.first; k = triple.second; v = triple.third
        } else {
            q = qProjection!!.forward(x)
            k = kProjection!!.forward(x)
            v = vProjection!!.forward(x)
        }

        // qk-norm: head_dim 단위 정규화 (RoPE 전). Q는 numHeads, K는 numKvHeads로 reshape.
        if (useQkNorm) {
            val qFlat = TurboTensor(intArrayOf(t * numHeads, headDim), q.data)
            val kFlat = TurboTensor(intArrayOf(t * numKvHeads, headDim), k.data)
            val qNormed = qNorm!!.forward(qFlat)
            val kNormed = kNorm!!.forward(kFlat)
            q = TurboTensor(intArrayOf(t, embedDim), qNormed.data)
            k = TurboTensor(intArrayOf(t, kvDim), kNormed.data)
        }

        if (useRoPE) {
            turboApplyRoPE(q, numHeads)
            turboApplyRoPE(k, numKvHeads)
        }
        cachedQ = q; cachedK = k; cachedV = v

        val attnProbs = FloatArray(numHeads * t * t)
        for (h in 0 until numHeads) {
            val kvHead = h / groupSize
            val qOffset = h * headDim
            val kvOffset = kvHead * headDim
            for (i in 0 until t) {
                val scoresRow = FloatArray(t)
                var maxScore = Float.NEGATIVE_INFINITY
                for (j in 0..i) {
                    var dot = 0.0f
                    for (d in 0 until headDim) {
                        dot += q[i, qOffset + d] * k[j, kvOffset + d]
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
            }
        }
        cachedAttnProbs = attnProbs

        val attnTensor = TurboTensor(intArrayOf(numHeads * t * t), attnProbs)
        val droppedAttnTensor = attnDropout.forward(attnTensor)
        val droppedAttn = droppedAttnTensor.data
        cachedDroppedAttn = droppedAttn

        val out = TurboTensor(intArrayOf(t, embedDim))
        for (h in 0 until numHeads) {
            val kvHead = h / groupSize
            val qOffset = h * headDim
            val kvOffset = kvHead * headDim
            for (i in 0 until t) {
                for (d in 0 until headDim) {
                    var acc = 0.0f
                    for (j in 0..i) {
                        acc += droppedAttn[h * t * t + i * t + j] * v[j, kvOffset + d]
                    }
                    out[i, qOffset + d] = acc
                }
            }
        }

        val projected = outputProjection.forward(out)
        return residDropout.forward(projected)
    }

    fun backward(gy: TurboTensor): TurboTensor {
        val q = cachedQ ?: error("forward 전 backward")
        val k = cachedK!!
        val v = cachedV!!
        val attn = cachedAttnProbs!!
        val droppedAttn = cachedDroppedAttn!!
        val t = cachedT

        val dResid = residDropout.backward(gy)
        val dMerged = outputProjection.backward(dResid)

        val dQ = TurboTensor(intArrayOf(t, embedDim))
        val dK = TurboTensor(intArrayOf(t, kvDim))
        val dV = TurboTensor(intArrayOf(t, kvDim))

        val dAttnDropped = FloatArray(numHeads * t * t)

        for (h in 0 until numHeads) {
            val kvHead = h / groupSize
            val qOffset = h * headDim
            val kvOffset = kvHead * headDim

            for (i in 0 until t) {
                for (j in 0..i) {
                    var acc = 0.0f
                    for (d in 0 until headDim) {
                        acc += dMerged[i, qOffset + d] * v[j, kvOffset + d]
                    }
                    dAttnDropped[h * t * t + i * t + j] = acc
                }
            }
            // GQA: 같은 kvHead를 공유하는 여러 h가 dV에 += 누적된다 (각 h가 독립적으로 contribution 추가).
            for (j in 0 until t) {
                for (d in 0 until headDim) {
                    var acc = 0.0f
                    for (i in j until t) {
                        acc += droppedAttn[h * t * t + i * t + j] * dMerged[i, qOffset + d]
                    }
                    dV[j, kvOffset + d] += acc
                }
            }
        }

        val dAttnAll = attnDropout.backward(TurboTensor(intArrayOf(numHeads * t * t), dAttnDropped)).data

        for (h in 0 until numHeads) {
            val kvHead = h / groupSize
            val qOffset = h * headDim
            val kvOffset = kvHead * headDim
            val dAttn = FloatArray(t * t)
            for (i in 0 until t) for (j in 0..i) {
                dAttn[i * t + j] = dAttnAll[h * t * t + i * t + j]
            }

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

            for (i in 0 until t) {
                for (d in 0 until headDim) {
                    var acc = 0.0f
                    for (j in 0..i) {
                        acc += dScores[i * t + j] * k[j, kvOffset + d]
                    }
                    dQ[i, qOffset + d] += scale * acc
                }
            }
            // GQA: 같은 kvHead를 공유하는 여러 h가 dK에 += 누적.
            for (j in 0 until t) {
                for (d in 0 until headDim) {
                    var acc = 0.0f
                    for (i in j until t) {
                        acc += dScores[i * t + j] * q[i, qOffset + d]
                    }
                    dK[j, kvOffset + d] += scale * acc
                }
            }
        }

        if (useRoPE) {
            turboApplyRoPEBackward(dQ, numHeads)
            turboApplyRoPEBackward(dK, numKvHeads)
        }

        // qk-norm backward — forward의 reshape 역순. Q는 numHeads, K는 numKvHeads.
        var dQOut: TurboTensor = dQ
        var dKOut: TurboTensor = dK
        if (useQkNorm) {
            val dQFlat = TurboTensor(intArrayOf(t * numHeads, headDim), dQ.data)
            val dKFlat = TurboTensor(intArrayOf(t * numKvHeads, headDim), dK.data)
            val dQAfter = qNorm!!.backward(dQFlat)
            val dKAfter = kNorm!!.backward(dKFlat)
            dQOut = TurboTensor(intArrayOf(t, embedDim), dQAfter.data)
            dKOut = TurboTensor(intArrayOf(t, kvDim), dKAfter.data)
        }

        if (useFusedQkv) {
            val dQkv = concatQkv(dQOut, dKOut, dV, embedDim, kvDim)
            return qkvProjection!!.backward(dQkv)
        } else {
            val dxQ = qProjection!!.backward(dQOut)
            val dxK = kProjection!!.backward(dKOut)
            val dxV = vProjection!!.backward(dV)

            val dx = TurboTensor(intArrayOf(t, embedDim))
            for (i in dx.data.indices) {
                dx.data[i] = dxQ.data[i] + dxK.data[i] + dxV.data[i]
            }
            return dx
        }
    }

    /**
     * KV cache incremental decode (Phase 3.0 — sampler 추론 가속).
     *   x: [1, embedDim] — 새 토큰 한 개의 입력 (이미 LayerNorm 거침)
     *   layer: KV cache의 layer 인덱스
     *   cache: 누적 K/V buffer
     *
     * 학습/backward 무관. dropout OFF (caller가 setTraining(false) 보장).
     *
     * 단순화 모드만 지원: numKvHeads=numHeads, !useFusedQkv, !useQkNorm.
     * 옵션 결합은 Phase 3.1+에서 확장.
     */
    fun forwardIncremental(x: TurboTensor, layer: Int, cache: TurboKVCache): TurboTensor {
        require(numKvHeads == numHeads) { "KV cache incremental: GQA 미지원 (Phase 3.1+)" }
        require(!useFusedQkv) { "KV cache incremental: fused QKV 미지원 (Phase 3.1+)" }
        require(!useQkNorm) { "KV cache incremental: qk-norm 미지원 (Phase 3.1+)" }
        require(x.rows == 1) { "incremental은 단일 토큰만 (rows=1), got ${x.rows}" }

        val qP = qProjection ?: error("qProjection null in non-fused mode")
        val kP = kProjection ?: error("kProjection null")
        val vP = vProjection ?: error("vProjection null")

        val qT = qP.forward(x)  // [1, embedDim]
        val kT = kP.forward(x)  // [1, kvDim]
        val vT = vP.forward(x)  // [1, kvDim]

        if (useRoPE) {
            turboApplyRoPEAtPosition(qT, numHeads, cache.currentPosition)
            turboApplyRoPEAtPosition(kT, numKvHeads, cache.currentPosition)
        }

        val seqLen = cache.append(layer, kT.data, vT.data)
        val kBuf = cache.getKBuffer(layer)
        val vBuf = cache.getVBuffer(layer)

        val out = TurboTensor(intArrayOf(1, embedDim))
        for (h in 0 until numHeads) {
            val kvHead = h / groupSize
            val qOffset = h * headDim
            val kvOffset = kvHead * headDim
            val scores = FloatArray(seqLen)
            var maxScore = Float.NEGATIVE_INFINITY
            for (j in 0 until seqLen) {
                var dot = 0.0f
                for (d in 0 until headDim) {
                    dot += qT.data[qOffset + d] * kBuf[j * kvDim + kvOffset + d]
                }
                val s = dot * scale
                scores[j] = s
                if (s > maxScore) maxScore = s
            }
            var sumExp = 0.0f
            for (j in 0 until seqLen) {
                val e = exp((scores[j] - maxScore).toDouble()).toFloat()
                scores[j] = e
                sumExp += e
            }
            val invSum = 1.0f / sumExp
            for (d in 0 until headDim) {
                var acc = 0.0f
                for (j in 0 until seqLen) {
                    acc += scores[j] * invSum * vBuf[j * kvDim + kvOffset + d]
                }
                out[0, qOffset + d] = acc
            }
        }

        return outputProjection.forward(out)
    }

    fun parameters(): List<TurboTensor> {
        val list = mutableListOf<TurboTensor>()
        if (useFusedQkv) {
            list += qkvProjection!!.parameters()
        } else {
            list += qProjection!!.parameters()
            list += kProjection!!.parameters()
            list += vProjection!!.parameters()
        }
        list += outputProjection.parameters()
        if (qNorm != null) list += qNorm.parameters()
        if (kNorm != null) list += kNorm.parameters()
        return list
    }

    /**
     * fused QKV 출력 [T, qDim+2*kvDim]를 (Q, K, V)로 분리.
     *   q ← cols [0, qDim)
     *   k ← cols [qDim, qDim+kvDim)
     *   v ← cols [qDim+kvDim, qDim+2*kvDim)
     */
    private fun sliceQkv(qkv: TurboTensor, qDim: Int, kvDim: Int): Triple<TurboTensor, TurboTensor, TurboTensor> {
        val t = qkv.rows
        val totalCols = qkv.cols
        require(totalCols == qDim + 2 * kvDim) {
            "fused QKV cols=$totalCols, expected ${qDim + 2 * kvDim}"
        }
        val q = TurboTensor(intArrayOf(t, qDim))
        val k = TurboTensor(intArrayOf(t, kvDim))
        val v = TurboTensor(intArrayOf(t, kvDim))
        for (i in 0 until t) {
            val src = i * totalCols
            val qDst = i * qDim
            val kDst = i * kvDim
            val vDst = i * kvDim
            for (j in 0 until qDim) q.data[qDst + j] = qkv.data[src + j]
            for (j in 0 until kvDim) k.data[kDst + j] = qkv.data[src + qDim + j]
            for (j in 0 until kvDim) v.data[vDst + j] = qkv.data[src + qDim + kvDim + j]
        }
        return Triple(q, k, v)
    }

    /** sliceQkv 역연산. dQ, dK, dV를 [T, qDim+2*kvDim]로 concat (sliceQkv의 backward). */
    private fun concatQkv(dQ: TurboTensor, dK: TurboTensor, dV: TurboTensor, qDim: Int, kvDim: Int): TurboTensor {
        val t = dQ.rows
        val totalCols = qDim + 2 * kvDim
        val out = TurboTensor(intArrayOf(t, totalCols))
        for (i in 0 until t) {
            val dst = i * totalCols
            val qSrc = i * qDim
            val kSrc = i * kvDim
            val vSrc = i * kvDim
            for (j in 0 until qDim) out.data[dst + j] = dQ.data[qSrc + j]
            for (j in 0 until kvDim) out.data[dst + qDim + j] = dK.data[kSrc + j]
            for (j in 0 until kvDim) out.data[dst + qDim + kvDim + j] = dV.data[vSrc + j]
        }
        return out
    }
}
