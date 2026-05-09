package turbo.layer

import turbo.TurboKVCache
import turbo.TurboModelConfig
import turbo.TurboTensor

/**
 * Pre-LN/Pre-RMSNorm Transformer 블록.
 *   h1 = x + attn(norm1(x))
 *   y  = h1 + mlp(norm2(h1))
 *
 * normalizationType, numKvHeads, useFusedQkv, useQkNorm 옵션은 TurboModelConfig에서 받음.
 * Phase 4에서 gradient checkpointing 토글 추가.
 */
class TurboTransformerBlock(modelConfig: TurboModelConfig) {
    private val gptConfig = modelConfig.gpt
    private val embedDim = gptConfig.embeddingDimension
    private val numHeads = gptConfig.numberOfAttentionHeads
    private val useBias = gptConfig.useBias
    private val dropoutProbability = gptConfig.dropoutProbability

    val layerNorm1: TurboNorm = createTurboNorm(embedDim, useBias, modelConfig.normalizationType)
    val attention: TurboSelfAttention = TurboSelfAttention(
        embedDim = embedDim,
        numHeads = numHeads,
        useBias = useBias,
        dropoutProbability = dropoutProbability,
        positionEncoding = modelConfig.positionEncoding,
        numKvHeads = modelConfig.effectiveKvHeads,
        useFusedQkv = modelConfig.useFusedQkv,
        useQkNorm = modelConfig.useQkNorm,
        normalizationType = modelConfig.normalizationType,
    )
    val layerNorm2: TurboNorm = createTurboNorm(embedDim, useBias, modelConfig.normalizationType)
    val mlp: TurboMLP = TurboMLP(embedDim, useBias, dropoutProbability, modelConfig.mlpActivation)

    /**
     * Phase 4.4 — gradient checkpointing 토글. true면 backward 시 forward 재실행으로
     * sub-layer cache 갱신 (메모리 vs 재계산 트레이드오프). dropout=0 권장 (mask 비결정성 회피).
     * 진짜 메모리 절감은 attention/mlp 내부 cache 명시 폐기까지 필요 — Phase 4.4.1+ 확장.
     */
    var useGradientCheckpointing: Boolean = false
    private var savedInput: TurboTensor? = null

    fun forward(x: TurboTensor): TurboTensor {
        if (useGradientCheckpointing) savedInput = x
        val attnOut = attention.forward(layerNorm1.forward(x))
        val h1 = addElementwise(x, attnOut)
        val mlpOut = mlp.forward(layerNorm2.forward(h1))
        return addElementwise(h1, mlpOut)
    }

    /** KV cache incremental forward — Phase 3.0. 학습/backward 무관. */
    fun forwardIncremental(x: TurboTensor, layer: Int, cache: TurboKVCache): TurboTensor {
        val attnOut = attention.forwardIncremental(layerNorm1.forward(x), layer, cache)
        val h1 = addElementwise(x, attnOut)
        val mlpOut = mlp.forward(layerNorm2.forward(h1))
        return addElementwise(h1, mlpOut)
    }

    fun backward(gy: TurboTensor): TurboTensor {
        if (useGradientCheckpointing) {
            // forward 재실행 — 모든 sub-layer cache 갱신 (deterministic w/ dropout=0)
            val saved = savedInput ?: error("gradient checkpointing forward 누락")
            val attnOut = attention.forward(layerNorm1.forward(saved))
            val h1 = addElementwise(saved, attnOut)
            mlp.forward(layerNorm2.forward(h1))
        }

        val dLn2Out = mlp.backward(gy)
        val dH1FromMlp = layerNorm2.backward(dLn2Out)
        val dH1 = addElementwise(gy, dH1FromMlp)

        val dLn1Out = attention.backward(dH1)
        val dXFromAttn = layerNorm1.backward(dLn1Out)
        return addElementwise(dH1, dXFromAttn)
    }

    fun parameters(): List<TurboTensor> =
        layerNorm1.parameters() +
                attention.parameters() +
                layerNorm2.parameters() +
                mlp.parameters()

    private fun addElementwise(a: TurboTensor, b: TurboTensor): TurboTensor {
        require(a.numel == b.numel)
        val out = TurboTensor(a.shape.copyOf())
        for (i in out.data.indices) out.data[i] = a.data[i] + b.data[i]
        return out
    }
}
