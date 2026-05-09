package turbo.layer

import turbo.TurboKVCache
import turbo.TurboModelConfig
import turbo.TurboTensor
import turbo.ops.turboMatmul
import turbo.transpose2D

/**
 * turbo 백엔드 GPT 모델. Phase 1에서 TurboModelConfig 기반 — RMSNorm/GQA/qk-norm/fused QKV/z-loss
 * 옵션을 받아 분기. 모든 옵션 OFF일 때 Phase 0 (vec 동등) 경로 그대로.
 */
class TurboPikoGPT(val config: TurboModelConfig) {
    private val gptConfig = config.gpt
    private val useRoPE: Boolean = config.positionEncoding.equals("rope", ignoreCase = true)

    val tokenEmbedding = TurboEmbeddingTable(gptConfig.vocabularySize, gptConfig.embeddingDimension)
    val positionEmbedding: TurboEmbeddingTable? =
        if (useRoPE) null else TurboEmbeddingTable(gptConfig.maxSequenceLength, gptConfig.embeddingDimension)
    val embeddingDropout = TurboDropout(gptConfig.dropoutProbability)
    val blocks: Array<TurboTransformerBlock> = Array(gptConfig.numberOfLayers) {
        TurboTransformerBlock(config)
    }
    val finalLayerNorm: TurboNorm =
        createTurboNorm(gptConfig.embeddingDimension, gptConfig.useBias, config.normalizationType)

    val lmHead: TurboLinear? =
        if (config.tieWeights) null
        else TurboLinear(gptConfig.embeddingDimension, gptConfig.vocabularySize, useBias = false)

    /** tied lm_head 모드(lmHead=null)에서만 backward에 필요한 finalLayerNorm 출력. */
    private var cachedHeadInput: TurboTensor? = null

    fun forward(tokenIds: IntArray): TurboTensor {
        val t = tokenIds.size
        require(t <= gptConfig.maxSequenceLength) { "시퀀스 길이 $t > maxSequenceLength ${gptConfig.maxSequenceLength}" }

        val tokEmb = tokenEmbedding.forward(tokenIds)
        val positionIds = IntArray(t) { it }
        var x = if (positionEmbedding != null) {
            val posEmb = positionEmbedding.forward(positionIds)
            addTensors(tokEmb, posEmb)
        } else {
            tokEmb
        }
        x = embeddingDropout.forward(x)

        for (block in blocks) {
            x = block.forward(x)
        }

        x = finalLayerNorm.forward(x)
        return if (lmHead != null) {
            lmHead.forward(x)
        } else {
            cachedHeadInput = x
            turboMatmul(x, tokenEmbedding.weight.transpose2D())
        }
    }

    /**
     * KV cache 기반 incremental forward (Phase 3.0).
     *   token: 다음에 처리할 토큰 id (단 한 개)
     *   cache: 모든 layer의 K/V 누적 buffer
     *
     * 호출자(sampler)는 setTraining(false) 보장. backward 안 함.
     * 단순 모드 (numKvHeads=numHeads, !useFusedQkv, !useQkNorm)만 지원.
     */
    fun forwardIncremental(token: Int, cache: TurboKVCache): TurboTensor {
        require(cache.numLayers == gptConfig.numberOfLayers) {
            "cache.numLayers=${cache.numLayers} != model layers=${gptConfig.numberOfLayers}"
        }
        require(cache.length < gptConfig.maxSequenceLength) {
            "position ${cache.length} >= maxSequenceLength ${gptConfig.maxSequenceLength}"
        }
        val tokEmb = tokenEmbedding.forward(intArrayOf(token))  // [1, embedDim]
        var x = if (positionEmbedding != null) {
            val posEmb = positionEmbedding.forward(intArrayOf(cache.length))
            addTensors(tokEmb, posEmb)
        } else {
            tokEmb
        }

        for ((i, block) in blocks.withIndex()) {
            x = block.forwardIncremental(x, i, cache)
        }

        x = finalLayerNorm.forward(x)
        return if (lmHead != null) {
            lmHead.forward(x)
        } else {
            turboMatmul(x, tokenEmbedding.weight.transpose2D())
        }
    }

    fun backward(gLogits: TurboTensor) {
        val dAfterLn = if (lmHead != null) {
            lmHead.backward(gLogits)
        } else {
            val x = cachedHeadInput ?: error("forward 없이 backward (tied head)")
            val w = tokenEmbedding.weight
            val dx = turboMatmul(gLogits, w)
            val wGrad = w.gradOrAlloc()
            val v = w.rows
            val c = w.cols
            val n = gLogits.rows
            for (vv in 0 until v) {
                for (cc in 0 until c) {
                    var sum = 0.0f
                    for (nn in 0 until n) {
                        sum += gLogits.data[nn * v + vv] * x.data[nn * c + cc]
                    }
                    wGrad[vv * c + cc] += sum
                }
            }
            dx
        }

        val dAfterBlocks = finalLayerNorm.backward(dAfterLn)

        var g = dAfterBlocks
        for (block in blocks.reversed()) {
            g = block.backward(g)
        }

        g = embeddingDropout.backward(g)

        tokenEmbedding.backward(g)
        positionEmbedding?.backward(g)
    }

    fun setTraining(mode: Boolean) {
        embeddingDropout.training = mode
        for (block in blocks) {
            block.attention.attnDropout.training = mode
            block.attention.residDropout.training = mode
            block.mlp.dropout.training = mode
        }
    }

    fun parameters(): List<TurboTensor> {
        val list = mutableListOf<TurboTensor>()
        list += tokenEmbedding.parameters()
        if (positionEmbedding != null) list += positionEmbedding.parameters()
        blocks.forEach { list += it.parameters() }
        list += finalLayerNorm.parameters()
        if (lmHead != null) list += lmHead.parameters()
        return list
    }

    fun zeroGrad() {
        parameters().forEach { it.zeroGrad() }
    }

    private fun addTensors(a: TurboTensor, b: TurboTensor): TurboTensor {
        require(a.numel == b.numel)
        val out = TurboTensor(a.shape.copyOf())
        for (i in out.data.indices) out.data[i] = a.data[i] + b.data[i]
        return out
    }
}
