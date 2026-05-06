package turbo

import gpt.GPTConfig
import turbo.layer.TurboPikoGPT
import turbo.ops.turboCrossEntropyForward
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Phase 1.2 — fused QKV 동등성.
 *
 * fused 모델의 qkvProjection weight를 unfused 모델의 qProj/kProj/vProj weight로 stacking하면
 * 두 모델은 수치적으로 동일한 forward 결과를 낸다.
 */
class TurboFusedQkvTest {

    private val gptCfg = GPTConfig(
        maxSequenceLength = 8,
        vocabularySize = 12,
        numberOfLayers = 2,
        numberOfAttentionHeads = 2,
        embeddingDimension = 8,
        useBias = true,
        dropoutProbability = 0.0f,
    )

    @Test
    fun fusedForwardMatchesUnfusedAfterWeightStacking() {
        val unfused = TurboPikoGPT(TurboModelConfig(gpt = gptCfg, useFusedQkv = false))
        val fused = TurboPikoGPT(TurboModelConfig(gpt = gptCfg, useFusedQkv = true))

        copyNonAttentionParams(fused, unfused)
        stackAttentionWeights(fused, unfused)

        val tokenIds = intArrayOf(0, 3, 5, 1, 4, 7, 2, 6)
        val uLogits = unfused.forward(tokenIds)
        val fLogits = fused.forward(tokenIds)

        var maxD = 0.0f
        for (i in uLogits.data.indices) {
            val d = abs(uLogits.data[i] - fLogits.data[i])
            if (d > maxD) maxD = d
        }
        assertTrue(maxD < 1e-4f, "fused vs unfused logits maxDiff=$maxD")
    }

    @Test
    fun fusedBackwardProducesGrads() {
        val model = TurboPikoGPT(TurboModelConfig(gpt = gptCfg, useFusedQkv = true))
        val tokenIds = intArrayOf(0, 3, 5, 1, 4, 7, 2, 6)
        val targets = intArrayOf(3, 5, 1, 4, 7, 2, 6, 0)

        val logits = model.forward(tokenIds)
        val ce = turboCrossEntropyForward(logits, targets)
        assertTrue(ce.loss.isFinite(), "loss NaN/Inf: ${ce.loss}")

        val gLogits = turbo.ops.turboCrossEntropyBackward(logits, targets, ce.softmax, 1.0f)
        model.backward(gLogits)

        val qkvW = model.blocks[0].attention.qkvProjection!!.weight
        var anyNonzero = false
        for (g in qkvW.grad ?: error("qkv grad alloc 안 됨")) {
            if (g != 0.0f) { anyNonzero = true; break }
        }
        assertTrue(anyNonzero, "qkv weight grad가 모두 0 — backward 누락 의심")
    }

    private fun copyNonAttentionParams(target: TurboPikoGPT, source: TurboPikoGPT) {
        // tokenEmbedding
        copyTensorList(target.tokenEmbedding.parameters(), source.tokenEmbedding.parameters())
        // positionEmbedding (RoPE면 둘 다 null)
        if (target.positionEmbedding != null && source.positionEmbedding != null) {
            copyTensorList(target.positionEmbedding!!.parameters(), source.positionEmbedding!!.parameters())
        }
        // 각 block의 ln1, ln2, mlp + attention.outputProjection
        for (i in target.blocks.indices) {
            val tBlk = target.blocks[i]
            val sBlk = source.blocks[i]
            copyTensorList(tBlk.layerNorm1.parameters(), sBlk.layerNorm1.parameters())
            copyTensorList(tBlk.layerNorm2.parameters(), sBlk.layerNorm2.parameters())
            copyTensorList(tBlk.mlp.parameters(), sBlk.mlp.parameters())
            // attention.outputProjection
            copyTensorList(
                tBlk.attention.outputProjection.parameters(),
                sBlk.attention.outputProjection.parameters(),
            )
        }
        copyTensorList(target.finalLayerNorm.parameters(), source.finalLayerNorm.parameters())
    }

    /** unfused.qProj/kProj/vProj weight + bias를 fused.qkvProjection에 [Q | K | V] 순으로 stacking. */
    private fun stackAttentionWeights(fused: TurboPikoGPT, unfused: TurboPikoGPT) {
        for (i in fused.blocks.indices) {
            val fAttn = fused.blocks[i].attention
            val uAttn = unfused.blocks[i].attention
            val embedDim = fAttn.embedDim
            val qkvProj = fAttn.qkvProjection ?: error("fused QKV 모드인데 qkvProjection null")
            val qProj = uAttn.qProjection ?: error("unfused 모드인데 qProjection null")
            val kProj = uAttn.kProjection ?: error("kProjection null")
            val vProj = uAttn.vProjection ?: error("vProjection null")
            val qkvW = qkvProj.weight
            val qW = qProj.weight
            val kW = kProj.weight
            val vW = vProj.weight
            for (r in 0 until embedDim) {
                for (c in 0 until embedDim) {
                    qkvW.data[r * embedDim + c] = qW.data[r * embedDim + c]
                    qkvW.data[(embedDim + r) * embedDim + c] = kW.data[r * embedDim + c]
                    qkvW.data[(2 * embedDim + r) * embedDim + c] = vW.data[r * embedDim + c]
                }
            }
            val qkvB = qkvProj.bias ?: error("qkv bias null")
            val qB = qProj.bias ?: error("q bias null")
            val kB = kProj.bias ?: error("k bias null")
            val vB = vProj.bias ?: error("v bias null")
            for (idx in 0 until embedDim) {
                qkvB.data[idx] = qB.data[idx]
                qkvB.data[embedDim + idx] = kB.data[idx]
                qkvB.data[2 * embedDim + idx] = vB.data[idx]
            }
        }
    }

    private fun copyTensorList(target: List<TurboTensor>, source: List<TurboTensor>) {
        require(target.size == source.size) { "param count mismatch: ${target.size} vs ${source.size}" }
        for (i in target.indices) {
            require(target[i].numel == source[i].numel)
            source[i].data.copyInto(target[i].data)
        }
    }
}
