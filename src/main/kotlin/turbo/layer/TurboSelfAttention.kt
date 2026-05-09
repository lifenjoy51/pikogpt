package turbo.layer

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import turbo.TurboKVCache
import turbo.TurboSimdMath
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

        // Phase A: Q·K^T inner d loop을 SIMD dot으로 가속.
        val qData = q.data
        val kData = k.data
        val attnProbs = FloatArray(numHeads * t * t)
        for (h in 0 until numHeads) {
            val kvHead = h / groupSize
            val qOffset = h * headDim
            val kvOffset = kvHead * headDim
            for (i in 0 until t) {
                val qRowOff = i * embedDim + qOffset
                val scoresRow = FloatArray(t)
                var maxScore = Float.NEGATIVE_INFINITY
                for (j in 0..i) {
                    val kRowOff = j * kvDim + kvOffset
                    val dot = TurboSimdMath.dot(qData, qRowOff, kData, kRowOff, headDim)
                    val s = dot * scale
                    scoresRow[j] = s
                    if (s > maxScore) maxScore = s
                }
                // Phase B: shifted exp + sum + normalize SIMD化 (j 범위 [0, i+1))
                val rowLen = i + 1
                val sSpecies = TurboSimdMath.SPECIES
                val sLane = sSpecies.length()
                val sUpper = sSpecies.loopBound(rowLen)
                val vMaxScore = FloatVector.broadcast(sSpecies, maxScore)
                var sumAcc = FloatVector.zero(sSpecies)
                var sj = 0
                while (sj < sUpper) {
                    val vS = FloatVector.fromArray(sSpecies, scoresRow, sj)
                    val vE = vS.sub(vMaxScore).lanewise(VectorOperators.EXP)
                    vE.intoArray(scoresRow, sj)
                    sumAcc = sumAcc.add(vE)
                    sj += sLane
                }
                var sumExp = sumAcc.reduceLanes(VectorOperators.ADD)
                while (sj < rowLen) {
                    val e = exp((scoresRow[sj] - maxScore).toDouble()).toFloat()
                    scoresRow[sj] = e
                    sumExp += e
                    sj++
                }
                val invSum = 1.0f / sumExp
                val attnBase = h * t * t + i * t
                val vInvSum = FloatVector.broadcast(sSpecies, invSum)
                sj = 0
                while (sj < sUpper) {
                    FloatVector.fromArray(sSpecies, scoresRow, sj).mul(vInvSum)
                        .intoArray(attnProbs, attnBase + sj)
                    sj += sLane
                }
                while (sj < rowLen) {
                    attnProbs[attnBase + sj] = scoresRow[sj] * invSum
                    sj++
                }
            }
        }
        cachedAttnProbs = attnProbs

        val attnTensor = TurboTensor(intArrayOf(numHeads * t * t), attnProbs)
        val droppedAttnTensor = attnDropout.forward(attnTensor)
        val droppedAttn = droppedAttnTensor.data
        cachedDroppedAttn = droppedAttn

        // Phase A: out_h[i, d] = Σ_j attn[i, j] * V[j, d] — d 차원 SIMD scaled add.
        val vData = v.data
        val out = TurboTensor(intArrayOf(t, embedDim))
        val outData = out.data
        val species = TurboSimdMath.SPECIES
        val laneLen = species.length()
        val dUpper = species.loopBound(headDim)
        val outRow = FloatArray(headDim)
        for (h in 0 until numHeads) {
            val kvHead = h / groupSize
            val qOffset = h * headDim
            val kvOffset = kvHead * headDim
            for (i in 0 until t) {
                java.util.Arrays.fill(outRow, 0.0f)
                val attnBase = h * t * t + i * t
                for (j in 0..i) {
                    val attnIj = droppedAttn[attnBase + j]
                    if (attnIj == 0.0f) continue
                    val vRowOff = j * kvDim + kvOffset
                    val vScalar = FloatVector.broadcast(species, attnIj)
                    var d = 0
                    while (d < dUpper) {
                        val vV = FloatVector.fromArray(species, vData, vRowOff + d)
                        val vO = FloatVector.fromArray(species, outRow, d)
                        vV.fma(vScalar, vO).intoArray(outRow, d)
                        d += laneLen
                    }
                    while (d < headDim) {
                        outRow[d] += attnIj * vData[vRowOff + d]
                        d++
                    }
                }
                val outRowOff = i * embedDim + qOffset
                System.arraycopy(outRow, 0, outData, outRowOff, headDim)
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
        val dMergedData = dMerged.data
        val vData2 = v.data
        val dVarr = dV.data
        val species = TurboSimdMath.SPECIES
        val laneLen = species.length()
        val dUpper = species.loopBound(headDim)

        for (h in 0 until numHeads) {
            val kvHead = h / groupSize
            val qOffset = h * headDim
            val kvOffset = kvHead * headDim

            // dAttnDropped[h, i, j] = Σ_d dMerged[i, qOff+d] * V[j, kvOff+d] — d 차원 SIMD dot
            for (i in 0 until t) {
                val dMergedRowOff = i * embedDim + qOffset
                val attnBase = h * t * t + i * t
                for (j in 0..i) {
                    val vRowOff = j * kvDim + kvOffset
                    dAttnDropped[attnBase + j] = TurboSimdMath.dot(
                        dMergedData, dMergedRowOff, vData2, vRowOff, headDim,
                    )
                }
            }
            // dV[j, kvOff+d] += Σ_i droppedAttn[h, i, j] * dMerged[i, qOff+d]
            //   j outer, i inner, d innermost SIMD scaled add (GQA에서 누적)
            for (j in 0 until t) {
                val dVRowOff = j * kvDim + kvOffset
                for (i in j until t) {
                    val attnIj = droppedAttn[h * t * t + i * t + j]
                    if (attnIj == 0.0f) continue
                    val vScalar = FloatVector.broadcast(species, attnIj)
                    val dMergedRowOff = i * embedDim + qOffset
                    var d = 0
                    while (d < dUpper) {
                        val vDM = FloatVector.fromArray(species, dMergedData, dMergedRowOff + d)
                        val vDV = FloatVector.fromArray(species, dVarr, dVRowOff + d)
                        vDM.fma(vScalar, vDV).intoArray(dVarr, dVRowOff + d)
                        d += laneLen
                    }
                    while (d < headDim) {
                        dVarr[dVRowOff + d] += attnIj * dMergedData[dMergedRowOff + d]
                        d++
                    }
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

            // dQ[i, qOff+d] += scale * Σ_j dScores[i, j] * K[j, kvOff+d] — d 차원 SIMD scaled add
            val kData2 = k.data
            val qData2 = q.data
            val dQData = dQ.data
            val dKData = dK.data
            for (i in 0 until t) {
                val dQRowOff = i * embedDim + qOffset
                for (j in 0..i) {
                    val ds = dScores[i * t + j] * scale
                    if (ds == 0.0f) continue
                    val kRowOff = j * kvDim + kvOffset
                    val vScalar = FloatVector.broadcast(species, ds)
                    var d = 0
                    while (d < dUpper) {
                        val vK = FloatVector.fromArray(species, kData2, kRowOff + d)
                        val vDQ = FloatVector.fromArray(species, dQData, dQRowOff + d)
                        vK.fma(vScalar, vDQ).intoArray(dQData, dQRowOff + d)
                        d += laneLen
                    }
                    while (d < headDim) {
                        dQData[dQRowOff + d] += ds * kData2[kRowOff + d]
                        d++
                    }
                }
            }
            // dK[j, kvOff+d] += scale * Σ_i dScores[i, j] * Q[i, qOff+d] (GQA 누적)
            for (j in 0 until t) {
                val dKRowOff = j * kvDim + kvOffset
                for (i in j until t) {
                    val ds = dScores[i * t + j] * scale
                    if (ds == 0.0f) continue
                    val qRowOff = i * embedDim + qOffset
                    val vScalar = FloatVector.broadcast(species, ds)
                    var d = 0
                    while (d < dUpper) {
                        val vQ = FloatVector.fromArray(species, qData2, qRowOff + d)
                        val vDK = FloatVector.fromArray(species, dKData, dKRowOff + d)
                        vQ.fma(vScalar, vDK).intoArray(dKData, dKRowOff + d)
                        d += laneLen
                    }
                    while (d < headDim) {
                        dKData[dKRowOff + d] += ds * qData2[qRowOff + d]
                        d++
                    }
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
