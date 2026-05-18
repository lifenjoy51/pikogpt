package mps

import org.junit.jupiter.api.Assumptions
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * P1.1 — batch 차원 일반화 검증.
 *
 *   1) 같은 token 시퀀스를 B번 복제했을 때 loss가 B=1과 거의 같다 (mean이 batch에서 평균됨).
 *   2) B=2와 B=1의 cache key가 다르다 — rebuild 트리거.
 */
class MpsGraphBatchTest {

    private fun ensure() {
        Assumptions.assumeTrue(MpsGraphSession.available(),
            "MpsGraph unavailable: ${MpsGraphSession.loadError}")
    }

    private fun makeFixture(seed: Long = 71L): Fixture {
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
        val tokens = IntArray(T) { rng.nextInt(vocab) }
        val targets = IntArray(T) { rng.nextInt(vocab) }
        return Fixture(T, vocab, numLayers, embedDim, numHeads, modelCfg, tm, tokens, targets, cos, sin, mask)
    }

    @Test
    fun b4ReplicatedMatchesB1Loss() {
        ensure()
        val fx = makeFixture()
        val cfgB1 = MpsGraphConfig(fx.numLayers, fx.embedDim, fx.numHeads, fx.T, fx.vocab, batchSize = 1)
        val cfgB4 = MpsGraphConfig(fx.numLayers, fx.embedDim, fx.numHeads, fx.T, fx.vocab, batchSize = 4)

        val lossB1: Float
        MpsGraphSession.create(cfgB1).use { s ->
            for ((idx, p) in fx.tm.parameters().withIndex()) s.loadWeights(idx, p.data, p.shape)
            lossB1 = s.runTrainingStep(fx.tokens, fx.targets, fx.cos, fx.sin, fx.mask,
                lr = 0f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f, weightDecay = 0f,
                stepT = 1, gradClip = 0f, batchSize = 1)
        }

        // 동일 시퀀스를 4번 복제. mean over batch는 동일 token이므로 결과도 동일해야.
        val tokensB4 = IntArray(4 * fx.T) { fx.tokens[it % fx.T] }
        val targetsB4 = IntArray(4 * fx.T) { fx.targets[it % fx.T] }

        val lossB4: Float
        MpsGraphSession.create(cfgB4).use { s ->
            for ((idx, p) in fx.tm.parameters().withIndex()) s.loadWeights(idx, p.data, p.shape)
            lossB4 = s.runTrainingStep(tokensB4, targetsB4, fx.cos, fx.sin, fx.mask,
                lr = 0f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f, weightDecay = 0f,
                stepT = 1, gradClip = 0f, batchSize = 4)
        }

        val diff = abs(lossB1 - lossB4)
        assertTrue(diff < 1e-3f,
            "B=1과 B=4 (동일 시퀀스 복제) loss 불일치: B1=$lossB1 B4=$lossB4 diff=$diff")
    }

    @Test
    fun differentSamplesInBatchProduceFiniteFiniteLoss() {
        ensure()
        val fx = makeFixture()
        val cfgB2 = MpsGraphConfig(fx.numLayers, fx.embedDim, fx.numHeads, fx.T, fx.vocab, batchSize = 2)

        val rng = Random(91L)
        val tokensB2 = IntArray(2 * fx.T) { rng.nextInt(fx.vocab) }
        val targetsB2 = IntArray(2 * fx.T) { rng.nextInt(fx.vocab) }

        MpsGraphSession.create(cfgB2).use { s ->
            for ((idx, p) in fx.tm.parameters().withIndex()) s.loadWeights(idx, p.data, p.shape)
            val loss = s.runTrainingStep(tokensB2, targetsB2, fx.cos, fx.sin, fx.mask,
                lr = 1e-2f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f, weightDecay = 0f,
                stepT = 1, gradClip = 0f, batchSize = 2)
            assertTrue(loss.isFinite() && !loss.isNaN(), "B=2 학습 loss invalid: $loss")
            assertTrue(loss > 0f, "loss는 양수여야: $loss")
        }
    }

    @Test
    fun forwardLossB4MatchesB1Replicated() {
        ensure()
        val fx = makeFixture()
        val cfg = MpsGraphConfig(fx.numLayers, fx.embedDim, fx.numHeads, fx.T, fx.vocab, batchSize = 4)
        val tokensB4 = IntArray(4 * fx.T) { fx.tokens[it % fx.T] }
        val targetsB4 = IntArray(4 * fx.T) { fx.targets[it % fx.T] }

        MpsGraphSession.create(cfg).use { s ->
            for ((idx, p) in fx.tm.parameters().withIndex()) s.loadWeights(idx, p.data, p.shape)
            val lossB4 = s.runForwardLoss(tokensB4, targetsB4, fx.cos, fx.sin, fx.mask, batchSize = 4)

            // B=1 동일 데이터
            val cfg1 = MpsGraphConfig(fx.numLayers, fx.embedDim, fx.numHeads, fx.T, fx.vocab, batchSize = 1)
            val lossB1: Float
            MpsGraphSession.create(cfg1).use { s2 ->
                for ((idx, p) in fx.tm.parameters().withIndex()) s2.loadWeights(idx, p.data, p.shape)
                lossB1 = s2.runForwardLoss(fx.tokens, fx.targets, fx.cos, fx.sin, fx.mask, batchSize = 1)
            }
            val diff = abs(lossB1 - lossB4)
            assertTrue(diff < 1e-3f,
                "forwardLoss B=1 vs B=4 (동일 시퀀스 복제) 불일치: B1=$lossB1 B4=$lossB4 diff=$diff")
        }
    }

    private data class Fixture(
        val T: Int, val vocab: Int, val numLayers: Int, val embedDim: Int, val numHeads: Int,
        val modelCfg: turbo.TurboModelConfig,
        val tm: turbo.layer.TurboPikoGPT,
        val tokens: IntArray, val targets: IntArray,
        val cos: FloatArray, val sin: FloatArray, val mask: FloatArray,
    )
}
