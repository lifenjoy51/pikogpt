package turbo

import gpt.GPTConfig
import turbo.layer.TurboPikoGPT
import turbo.ops.turboCrossEntropyBackward
import turbo.ops.turboCrossEntropyForward
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Phase 1.3 — GQA (Grouped Query Attention) 검증.
 *
 *   numKvHeads = null (= numHeads) → MHA, kvDim = embedDim
 *   numKvHeads < numHeads → GQA, kvDim = numKvHeads * headDim < embedDim
 */
class TurboGqaTest {

    private val gptCfg = GPTConfig(
        maxSequenceLength = 8,
        vocabularySize = 12,
        numberOfLayers = 2,
        numberOfAttentionHeads = 4,
        embeddingDimension = 16,
        useBias = true,
        dropoutProbability = 0.0f,
    )

    @Test
    fun gqaReducesKvProjectionParams() {
        val mha = TurboPikoGPT(TurboModelConfig(gpt = gptCfg, numKvHeads = null))
        val gqa = TurboPikoGPT(TurboModelConfig(gpt = gptCfg, numKvHeads = 2))   // 4 Q heads, 2 KV heads
        val mqa = TurboPikoGPT(TurboModelConfig(gpt = gptCfg, numKvHeads = 1))   // MQA 극단

        val mhaCount = mha.parameters().sumOf { it.numel }
        val gqaCount = gqa.parameters().sumOf { it.numel }
        val mqaCount = mqa.parameters().sumOf { it.numel }

        assertTrue(gqaCount < mhaCount, "GQA 파라미터 수가 MHA보다 작아야: gqa=$gqaCount, mha=$mhaCount")
        assertTrue(mqaCount < gqaCount, "MQA 파라미터 수가 GQA보다 작아야: mqa=$mqaCount, gqa=$gqaCount")

        // 정확한 절감량: MHA의 K, V projection은 [embedDim, embedDim] 각각.
        // GQA(numKvHeads=2)는 [embedDim, kvDim=8] — 두 layer × 2개 (K,V) × (16-8) col × 16 row = 1024
        //   + bias 절감: 두 layer × 2개 × (16-8) = 32
        //   = 1056
        val savedPerHead = (gptCfg.numberOfAttentionHeads - 2) * gptCfg.embeddingDimension / gptCfg.numberOfAttentionHeads
        val expectedSavedW = gptCfg.numberOfLayers * 2 * savedPerHead * gptCfg.embeddingDimension
        val expectedSavedB = gptCfg.numberOfLayers * 2 * savedPerHead
        val expectedDelta = expectedSavedW + expectedSavedB
        assertEquals(expectedDelta, mhaCount - gqaCount, "GQA 파라미터 절감량 mismatch")
    }

    @Test
    fun gqaForwardBackwardRunsWithoutNaN() {
        val model = TurboPikoGPT(TurboModelConfig(gpt = gptCfg, numKvHeads = 2))
        val tokenIds = intArrayOf(0, 3, 5, 1, 4, 7, 2, 6)
        val targets = intArrayOf(3, 5, 1, 4, 7, 2, 6, 0)

        val logits = model.forward(tokenIds)
        for (v in logits.data) assertTrue(v.isFinite(), "logits NaN/Inf: $v")

        val ce = turboCrossEntropyForward(logits, targets)
        assertTrue(ce.loss.isFinite(), "loss NaN/Inf: ${ce.loss}")

        val gLogits = turboCrossEntropyBackward(logits, targets, ce.softmax, 1.0f)
        model.backward(gLogits)

        // K/V projection이 [embedDim, kvDim] 차원이고 grad 누적 확인
        val kProj = model.blocks[0].attention.kProjection ?: error("kProjection null")
        val vProj = model.blocks[0].attention.vProjection ?: error("vProjection null")
        assertEquals(8, kProj.outFeatures, "GQA kProjection out_features = kvDim")
        assertEquals(8, vProj.outFeatures, "GQA vProjection out_features = kvDim")
        val kGrad = kProj.weight.grad ?: error("kProj grad alloc 안 됨")
        val vGrad = vProj.weight.grad ?: error("vProj grad alloc 안 됨")
        var anyK = false; for (g in kGrad) if (g != 0.0f) { anyK = true; break }
        var anyV = false; for (g in vGrad) if (g != 0.0f) { anyV = true; break }
        assertTrue(anyK, "kProj grad 모두 0 — backward 누락 의심")
        assertTrue(anyV, "vProj grad 모두 0 — backward 누락 의심")
    }

    @Test
    fun gqaPlusFusedQkvWorks() {
        val model = TurboPikoGPT(TurboModelConfig(
            gpt = gptCfg,
            numKvHeads = 2,
            useFusedQkv = true,
        ))
        val tokenIds = intArrayOf(0, 3, 5, 1, 4, 7, 2, 6)
        val targets = intArrayOf(3, 5, 1, 4, 7, 2, 6, 0)

        val logits = model.forward(tokenIds)
        for (v in logits.data) assertTrue(v.isFinite(), "logits NaN/Inf: $v")
        val ce = turboCrossEntropyForward(logits, targets)
        val gLogits = turboCrossEntropyBackward(logits, targets, ce.softmax, 1.0f)
        model.backward(gLogits)

        val qkvProj = model.blocks[0].attention.qkvProjection ?: error("qkvProj null")
        // out = embedDim + 2*kvDim = 16 + 2*8 = 32
        assertEquals(32, qkvProj.outFeatures, "fused QKV out = embedDim + 2*kvDim")
        val grad = qkvProj.weight.grad ?: error("qkv grad alloc 안 됨")
        var any = false; for (g in grad) if (g != 0.0f) { any = true; break }
        assertTrue(any, "qkv grad 모두 0 — backward 누락")
    }
}
