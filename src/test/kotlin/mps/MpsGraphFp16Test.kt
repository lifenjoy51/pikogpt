package mps

import org.junit.jupiter.api.Assumptions
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * P2.1 — fp16 mixed precision 검증.
 *
 *   1) useFp16=true 모드에서 학습이 정상 진행 (loss 감소)
 *   2) fp16 forward의 loss가 fp32 baseline 대비 ±0.05 이내 (RoPE/SwiGLU/LN 누적 정밀도 손해)
 */
class MpsGraphFp16Test {

    private fun ensure() {
        Assumptions.assumeTrue(MpsGraphSession.available(),
            "MpsGraph unavailable: ${MpsGraphSession.loadError}")
    }

    private fun fx(seed: Long, useFp16: Boolean): Fx {
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
            gpt = gptCfg, tieWeights = true, mlpActivation = "swiglu",
            positionEncoding = "rope", normalizationType = "layernorm",
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
            useFp16 = useFp16,
        )
        val tokens = IntArray(T) { Random(seed + 1).nextInt(vocab) }
        val targets = IntArray(T) { Random(seed + 2).nextInt(vocab) }
        return Fx(cfg, modelCfg, tm, T, vocab, cos, sin, mask, tokens, targets)
    }

    @Test
    fun fp16ModeTrainsAndReducesLoss() {
        ensure()
        val f = fx(seed = 31L, useFp16 = true)
        MpsGraphSession.create(f.cfg).use { s ->
            for ((idx, p) in f.tm.parameters().withIndex()) s.loadWeights(idx, p.data, p.shape)

            val firstLoss = s.runTrainingStep(
                f.tokens, f.targets, f.cos, f.sin, f.mask,
                lr = 0f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f, weightDecay = 0f,
                stepT = 1, gradClip = 0f, batchSize = 1,
            )
            var lastLoss = 0f
            for (it in 2..30) {
                lastLoss = s.runTrainingStep(
                    f.tokens, f.targets, f.cos, f.sin, f.mask,
                    lr = 1e-2f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f, weightDecay = 0f,
                    stepT = it, gradClip = 0f, batchSize = 1,
                )
            }
            assertTrue(lastLoss.isFinite() && !lastLoss.isNaN(),
                "fp16 학습 loss invalid: $lastLoss")
            assertTrue(lastLoss < firstLoss - 0.3f,
                "fp16 mode 학습이 loss를 줄이지 못함: first=$firstLoss last=$lastLoss")
        }
    }

    @Test
    fun fp16ForwardCloseToFp32Forward() {
        ensure()
        val seed = 41L
        val ff32 = fx(seed = seed, useFp16 = false)
        val ff16 = fx(seed = seed, useFp16 = true)

        val l32: Float
        MpsGraphSession.create(ff32.cfg).use { s ->
            for ((idx, p) in ff32.tm.parameters().withIndex()) s.loadWeights(idx, p.data, p.shape)
            l32 = s.runTrainingStep(
                ff32.tokens, ff32.targets, ff32.cos, ff32.sin, ff32.mask,
                lr = 0f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f, weightDecay = 0f,
                stepT = 1, gradClip = 0f, batchSize = 1,
            )
        }
        val l16: Float
        MpsGraphSession.create(ff16.cfg).use { s ->
            for ((idx, p) in ff16.tm.parameters().withIndex()) s.loadWeights(idx, p.data, p.shape)
            l16 = s.runTrainingStep(
                ff16.tokens, ff16.targets, ff16.cos, ff16.sin, ff16.mask,
                lr = 0f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f, weightDecay = 0f,
                stepT = 1, gradClip = 0f, batchSize = 1,
            )
        }
        val diff = abs(l32 - l16)
        assertTrue(l16.isFinite() && !l16.isNaN(), "fp16 forward loss invalid: $l16")
        assertTrue(diff < 0.05f,
            "fp16 forward가 fp32와 너무 다름: l32=$l32 l16=$l16 diff=$diff")
    }

    private data class Fx(
        val cfg: MpsGraphConfig, val modelCfg: turbo.TurboModelConfig,
        val tm: turbo.layer.TurboPikoGPT,
        val T: Int, val vocab: Int,
        val cos: FloatArray, val sin: FloatArray, val mask: FloatArray,
        val tokens: IntArray, val targets: IntArray,
    )
}
