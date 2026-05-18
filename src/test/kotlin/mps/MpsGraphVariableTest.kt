package mps

import org.junit.jupiter.api.Assumptions
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * P1.3 — Variable 패러다임 검증.
 *
 *   1) useVariableForStep=true 모드에서 학습이 정상 진행 (loss 감소)
 *   2) variable mode와 placeholder mode가 동일 seed/data로 같은 학습 결과 (rtol < 1e-4)
 *
 * variable mode 의도:
 *   - stepGraph 안에서 weight/m/v를 MPSGraph variable로 보관 (placeholder 대신)
 *   - assignVariable로 in-place update (graph 내부)
 *   - 매 step weight feeds 생략 (graph variable이 자체 storage)
 *   - functional 검증: 두 mode의 weight update가 수학적으로 동일해야
 */
class MpsGraphVariableTest {

    private fun ensure() {
        Assumptions.assumeTrue(MpsGraphSession.available(),
            "MpsGraph unavailable: ${MpsGraphSession.loadError}")
    }

    private fun fx(seed: Long = 13L, useVariable: Boolean): Fx {
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
            useVariableForStep = useVariable,
        )
        val tokens = IntArray(T) { Random(seed + 1).nextInt(vocab) }
        val targets = IntArray(T) { Random(seed + 2).nextInt(vocab) }
        return Fx(cfg, modelCfg, tm, T, vocab, cos, sin, mask, tokens, targets)
    }

    @Test
    fun variableModeTrainsAndReducesLoss() {
        ensure()
        val f = fx(useVariable = true)
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
                "variable mode 학습이 loss를 줄이지 못함: first=$firstLoss last=$lastLoss")
        }
    }

    @Test
    fun variableModeMatchesPlaceholderMode() {
        ensure()
        // 두 mode에서 동일 weight init + 동일 step sequence → 같은 weight update.
        // 검증: readWeight로 두 mode 최종 weight L2 diff < 1e-4.
        val seed = 23L
        val fp = fx(seed = seed, useVariable = false)
        val fv = fx(seed = seed, useVariable = true)

        val pWeights = mutableListOf<FloatArray>()
        val vWeights = mutableListOf<FloatArray>()

        MpsGraphSession.create(fp.cfg).use { s ->
            for ((idx, p) in fp.tm.parameters().withIndex()) s.loadWeights(idx, p.data, p.shape)
            for (it in 1..5) {
                s.runTrainingStep(
                    fp.tokens, fp.targets, fp.cos, fp.sin, fp.mask,
                    lr = 1e-2f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f, weightDecay = 0f,
                    stepT = it, gradClip = 0f, batchSize = 1,
                )
            }
            for ((idx, p) in fp.tm.parameters().withIndex()) {
                val w = FloatArray(p.numel)
                s.readWeight(idx, w)
                pWeights.add(w)
            }
        }

        MpsGraphSession.create(fv.cfg).use { s ->
            for ((idx, p) in fv.tm.parameters().withIndex()) s.loadWeights(idx, p.data, p.shape)
            for (it in 1..5) {
                s.runTrainingStep(
                    fv.tokens, fv.targets, fv.cos, fv.sin, fv.mask,
                    lr = 1e-2f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f, weightDecay = 0f,
                    stepT = it, gradClip = 0f, batchSize = 1,
                )
            }
            for ((idx, p) in fv.tm.parameters().withIndex()) {
                val w = FloatArray(p.numel)
                s.readWeight(idx, w)
                vWeights.add(w)
            }
        }

        var totalDiff = 0.0
        var totalNorm = 0.0
        for (i in pWeights.indices) {
            for (j in pWeights[i].indices) {
                val d = (pWeights[i][j] - vWeights[i][j]).toDouble()
                totalDiff += d * d
                totalNorm += pWeights[i][j].toDouble() * pWeights[i][j].toDouble()
            }
        }
        val l2Diff = kotlin.math.sqrt(totalDiff)
        val l2Norm = kotlin.math.sqrt(totalNorm)
        val relDiff = l2Diff / l2Norm
        assertTrue(relDiff < 1e-4,
            "variable vs placeholder mode 결과 불일치: l2_diff=$l2Diff, l2_norm=$l2Norm, rel=$relDiff")
    }

    private data class Fx(
        val cfg: MpsGraphConfig, val modelCfg: turbo.TurboModelConfig,
        val tm: turbo.layer.TurboPikoGPT,
        val T: Int, val vocab: Int,
        val cos: FloatArray, val sin: FloatArray, val mask: FloatArray,
        val tokens: IntArray, val targets: IntArray,
    )
}
