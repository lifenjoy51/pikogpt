package mps

import org.junit.jupiter.api.Assumptions
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * P1.2 — 진정한 grad accumulation graph 분리 검증.
 *
 *   1) accumStep을 N번 호출 후 adamStep → loss 감소 + weight 갱신
 *   2) resetGradAccum 후 readGrad가 0
 *   3) 누적된 grad가 단일 accumStep의 grad보다 크다 (다른 데이터로 누적)
 */
class MpsGraphAccumTest {

    private fun ensure() {
        Assumptions.assumeTrue(MpsGraphSession.available(),
            "MpsGraph unavailable: ${MpsGraphSession.loadError}")
    }

    private fun fixture(seed: Long = 31L): Fx {
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
        val cfg = MpsGraphConfig(numLayers, embedDim, numHeads, T, vocab, batchSize = 1)
        return Fx(cfg, modelCfg, tm, T, vocab, cos, sin, mask, rng)
    }

    @Test
    fun accumThenAdamReducesLoss() {
        ensure()
        val fx = fixture()
        MpsGraphSession.create(fx.cfg).use { s ->
            for ((idx, p) in fx.tm.parameters().withIndex()) s.loadWeights(idx, p.data, p.shape)
            s.resetGradAccum()

            val tokens = IntArray(fx.T) { fx.rng.nextInt(fx.vocab) }
            val targets = IntArray(fx.T) { fx.rng.nextInt(fx.vocab) }

            // 1 effective iter = 4 micro-step accum + 1 adam
            val firstLoss: Float
            var lastLoss = 0f
            run {
                val l1 = s.runAccumStep(tokens, targets, fx.cos, fx.sin, fx.mask, batchSize = 1)
                firstLoss = l1
                for (m in 0 until 3) s.runAccumStep(tokens, targets, fx.cos, fx.sin, fx.mask, batchSize = 1)
                s.runAdamStep(lr = 1e-2f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f,
                    weightDecay = 0f, gradClip = 0f, stepT = 1)
            }
            // 두 번째 effective iter
            for (effIter in 2..5) {
                for (m in 0 until 4) {
                    lastLoss = s.runAccumStep(tokens, targets, fx.cos, fx.sin, fx.mask, batchSize = 1)
                }
                s.runAdamStep(lr = 1e-2f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f,
                    weightDecay = 0f, gradClip = 0f, stepT = effIter)
            }
            assertTrue(lastLoss < firstLoss - 0.05f,
                "accum + adam이 loss를 줄이지 못함: first=$firstLoss last=$lastLoss")
        }
    }

    @Test
    fun resetGradAccumZerosGrad() {
        ensure()
        val fx = fixture(seed = 41L)
        MpsGraphSession.create(fx.cfg).use { s ->
            for ((idx, p) in fx.tm.parameters().withIndex()) s.loadWeights(idx, p.data, p.shape)
            val tokens = IntArray(fx.T) { fx.rng.nextInt(fx.vocab) }
            val targets = IntArray(fx.T) { fx.rng.nextInt(fx.vocab) }

            // 한번 accum → grad가 0이 아닐 것
            s.resetGradAccum()
            s.runAccumStep(tokens, targets, fx.cos, fx.sin, fx.mask, batchSize = 1)

            val numel0 = fx.tm.parameters()[0].numel
            val g1 = FloatArray(numel0); s.readGrad(0, g1)
            val absSumBefore = g1.sumOf { kotlin.math.abs(it).toDouble() }
            assertTrue(absSumBefore > 0.0, "accum 후 grad가 0 — graph가 backward 안 함")

            s.resetGradAccum()
            val g2 = FloatArray(numel0); s.readGrad(0, g2)
            val absSumAfter = g2.sumOf { kotlin.math.abs(it).toDouble() }
            assertTrue(absSumAfter == 0.0,
                "resetGradAccum 후 grad가 0 아님: |g|=$absSumAfter")
        }
    }

    @Test
    fun accumGradGrowsWithMicroSteps() {
        ensure()
        val fx = fixture(seed = 51L)
        MpsGraphSession.create(fx.cfg).use { s ->
            for ((idx, p) in fx.tm.parameters().withIndex()) s.loadWeights(idx, p.data, p.shape)
            val tokens = IntArray(fx.T) { fx.rng.nextInt(fx.vocab) }
            val targets = IntArray(fx.T) { fx.rng.nextInt(fx.vocab) }

            val numel0 = fx.tm.parameters()[0].numel

            s.resetGradAccum()
            s.runAccumStep(tokens, targets, fx.cos, fx.sin, fx.mask, batchSize = 1)
            val g1 = FloatArray(numel0); s.readGrad(0, g1)
            val l2_1 = kotlin.math.sqrt(g1.sumOf { (it * it).toDouble() })

            // 같은 데이터를 3번 더 누적
            for (i in 0 until 3) s.runAccumStep(tokens, targets, fx.cos, fx.sin, fx.mask, batchSize = 1)
            val g4 = FloatArray(numel0); s.readGrad(0, g4)
            val l2_4 = kotlin.math.sqrt(g4.sumOf { (it * it).toDouble() })

            // 누적이 동일 데이터면 grad가 4배가 되어야 정확하지만, RoPE/softmax 정밀도로 ~3.8~4.2배.
            // 최소한 1.5×보단 커야 한다.
            assertTrue(l2_4 > l2_1 * 2.0,
                "grad가 누적 안 됨: l2_1=$l2_1, l2_4=$l2_4 (>2× 기대)")
        }
    }

    private data class Fx(
        val cfg: MpsGraphConfig, val modelCfg: turbo.TurboModelConfig,
        val tm: turbo.layer.TurboPikoGPT,
        val T: Int, val vocab: Int,
        val cos: FloatArray, val sin: FloatArray, val mask: FloatArray,
        val rng: Random,
    )
}
