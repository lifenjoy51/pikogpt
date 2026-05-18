package mps

import org.junit.jupiter.api.Assumptions
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * P4 — useSwiglu=false (GELU MLP) 회귀.
 *
 * 모델 1-layer, 8 token, GELU activation, RoPE PE (PE는 본 test 관심 밖이라 기본).
 * 검증:
 *   1) GELU graph build + forward 통과 (loss finite)
 *   2) 20 step 학습 후 loss 감소 (학습 메커니즘 자체 동작)
 */
class MpsGraphGeluTest {

    private fun ensure() {
        Assumptions.assumeTrue(MpsGraphSession.available(),
            "MpsGraph unavailable: ${MpsGraphSession.loadError}")
    }

    private data class Fx(
        val cfg: MpsGraphConfig,
        val tm: turbo.layer.TurboPikoGPT,
        val T: Int, val cos: FloatArray, val sin: FloatArray, val mask: FloatArray,
        val tokens: IntArray, val targets: IntArray,
    )

    private fun fx(seed: Long = 17L): Fx {
        val T = 8
        val embedDim = 16
        val numHeads = 4
        val numLayers = 1
        val vocab = 32
        val headDim = embedDim / numHeads
        val half = headDim / 2

        val gptCfg = gpt.GPTConfig(
            vocabularySize = vocab, embeddingDimension = embedDim,
            numberOfLayers = numLayers, numberOfAttentionHeads = numHeads,
            maxSequenceLength = T, dropoutProbability = 0.0f, useBias = true,
        )
        val modelCfg = turbo.TurboModelConfig(
            gpt = gptCfg, tieWeights = true,
            mlpActivation = "gelu",          // ← GELU 분기 활성화
            positionEncoding = "rope",       // PE는 RoPE 그대로
            normalizationType = "layernorm",
        )
        val tm = turbo.layer.TurboPikoGPT(modelCfg)
        val rng = Random(seed)
        for (p in tm.parameters()) for (i in 0 until p.numel) p.data[i] = (rng.nextFloat() - 0.5f) * 0.2f

        val cos = FloatArray(T * half)
        val sin = FloatArray(T * half)
        for (t in 0 until T) for (i in 0 until half) {
            val theta = Math.pow(10000.0, -2.0 * i / headDim)
            cos[t * half + i] = kotlin.math.cos(t * theta).toFloat()
            sin[t * half + i] = kotlin.math.sin(t * theta).toFloat()
        }
        val mask = FloatArray(T * T) { idx -> if (idx % T > idx / T) -1e9f else 0f }
        val cfg = MpsGraphConfig(
            numLayers, embedDim, numHeads, T, vocab, batchSize = 1,
            useSwiglu = false, useRope = true,
        )
        val tokens = IntArray(T) { Random(seed + 1).nextInt(vocab) }
        val targets = IntArray(T) { Random(seed + 2).nextInt(vocab) }
        return Fx(cfg, tm, T, cos, sin, mask, tokens, targets)
    }

    @Test
    fun geluForwardProducesFiniteLoss() {
        ensure()
        val f = fx()
        MpsGraphSession.create(f.cfg).use { s ->
            for ((idx, p) in f.tm.parameters().withIndex()) s.loadWeights(idx, p.data, p.shape)
            val loss = s.runForwardLoss(f.tokens, f.targets, f.cos, f.sin, f.mask, batchSize = 1)
            assertTrue(loss.isFinite(), "GELU forward loss not finite: $loss")
            assertTrue(loss > 0f, "GELU forward loss <= 0: $loss")
        }
    }

    @Test
    fun geluTrainsAndReducesLoss() {
        ensure()
        val f = fx()
        MpsGraphSession.create(f.cfg).use { s ->
            for ((idx, p) in f.tm.parameters().withIndex()) s.loadWeights(idx, p.data, p.shape)
            val firstLoss = s.runTrainingStep(
                f.tokens, f.targets, f.cos, f.sin, f.mask,
                lr = 0f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f, weightDecay = 0f,
                stepT = 1, gradClip = 0f, batchSize = 1,
            )
            var lastLoss = 0f
            for (it in 2..20) {
                lastLoss = s.runTrainingStep(
                    f.tokens, f.targets, f.cos, f.sin, f.mask,
                    lr = 1e-2f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f, weightDecay = 0f,
                    stepT = it, gradClip = 0f, batchSize = 1,
                )
            }
            assertTrue(lastLoss < firstLoss - 0.5f,
                "GELU 학습이 loss 줄이지 못함: first=$firstLoss last=$lastLoss")
        }
    }
}
