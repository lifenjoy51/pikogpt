package mps

import org.junit.jupiter.api.Assumptions
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * P4 — useDropout 회귀.
 *
 *   1) useDropout=true + dropoutMask=null 시 mask=1 강제 → useDropout=false와 같은 loss
 *      (graph는 dropout op 있지만 mask=1이라 identity)
 *   2) useDropout=true + 무작위 mask로 학습 진행 시 loss 감소 (graph가 정상 backward)
 */
class MpsGraphDropoutTest {

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

    private fun fx(useDropout: Boolean, p: Float, seed: Long = 41L): Fx {
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
            mlpActivation = "swiglu", positionEncoding = "rope",
            normalizationType = "layernorm",
        )
        val tm = turbo.layer.TurboPikoGPT(modelCfg)
        val rng = Random(seed)
        for (param in tm.parameters()) for (i in 0 until param.numel) param.data[i] = (rng.nextFloat() - 0.5f) * 0.2f

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
            useDropout = useDropout, dropoutProbability = p,
        )
        val tokens = IntArray(T) { Random(seed + 1).nextInt(vocab) }
        val targets = IntArray(T) { Random(seed + 2).nextInt(vocab) }
        return Fx(cfg, tm, T, cos, sin, mask, tokens, targets)
    }

    @Test
    fun dropoutMaskNullEqualsNoDropout() {
        ensure()
        // 같은 seed → 같은 weight init. useDropout=true mask=null이면 host에서 mask=1 강제 → identity.
        val fOn  = fx(useDropout = true,  p = 0.5f)
        val fOff = fx(useDropout = false, p = 0.0f)
        val lOn = MpsGraphSession.create(fOn.cfg).use { s ->
            for ((idx, p) in fOn.tm.parameters().withIndex()) s.loadWeights(idx, p.data, p.shape)
            s.runTrainingStep(
                fOn.tokens, fOn.targets, fOn.cos, fOn.sin, fOn.mask,
                lr = 0f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f, weightDecay = 0f,
                stepT = 1, gradClip = 0f, batchSize = 1, dropoutMask = null,
            )
        }
        val lOff = MpsGraphSession.create(fOff.cfg).use { s ->
            for ((idx, p) in fOff.tm.parameters().withIndex()) s.loadWeights(idx, p.data, p.shape)
            s.runTrainingStep(
                fOff.tokens, fOff.targets, fOff.cos, fOff.sin, fOff.mask,
                lr = 0f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f, weightDecay = 0f,
                stepT = 1, gradClip = 0f, batchSize = 1,
            )
        }
        val diff = abs(lOn - lOff)
        assertTrue(diff < 1e-4f,
            "dropout mask=1 vs dropout off loss diff=$diff (≥ 1e-4): lOn=$lOn lOff=$lOff")
    }

    @Test
    fun dropoutTrainsAndReducesLoss() {
        ensure()
        val f = fx(useDropout = true, p = 0.1f)
        MpsGraphSession.create(f.cfg).use { s ->
            for ((idx, p) in f.tm.parameters().withIndex()) s.loadWeights(idx, p.data, p.shape)
            // 매 step 새 random mask 생성 (T*embedDim*2 layer = 8*16*2*1 = 256).
            val maskSize = 2 * f.cfg.numLayers * f.cfg.batchSize * f.T * f.cfg.embedDim
            val keep = 1.0f / (1.0f - 0.1f)
            val rng = Random(7L)
            fun makeMask(): FloatArray {
                val m = FloatArray(maskSize)
                for (i in 0 until maskSize) m[i] = if (rng.nextFloat() < 0.1f) 0.0f else keep
                return m
            }
            val firstLoss = s.runTrainingStep(
                f.tokens, f.targets, f.cos, f.sin, f.mask,
                lr = 0f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f, weightDecay = 0f,
                stepT = 1, gradClip = 0f, batchSize = 1, dropoutMask = makeMask(),
            )
            var lastLoss = 0f
            for (it in 2..30) {
                lastLoss = s.runTrainingStep(
                    f.tokens, f.targets, f.cos, f.sin, f.mask,
                    lr = 1e-2f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f, weightDecay = 0f,
                    stepT = it, gradClip = 0f, batchSize = 1, dropoutMask = makeMask(),
                )
            }
            assertTrue(lastLoss < firstLoss - 0.3f,
                "dropout 학습이 loss 줄이지 못함: first=$firstLoss last=$lastLoss")
        }
    }
}
