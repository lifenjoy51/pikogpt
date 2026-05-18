package mps

import org.junit.jupiter.api.Assumptions
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * P0.4 — Global gradient norm clipping이 step graph에 들어갔는지 검증.
 *
 *   1) clip disabled(0)와 매우 큰 clip(1e10)이 같은 loss를 produce — clipping op가 no-op 처리 OK
 *   2) 5 step 학습 후 weight 변화량이 큰 clip vs 작은 clip 사이에 비례적으로 다르다 — clipping이
 *      누적 update에 실제로 영향 (AdamW는 첫 step에 sign-only update여서 multi-step이 필요)
 *   3) clipping이 NaN/Inf 발생시키지 않음
 */
class MpsGraphGradClipTest {

    private fun ensure() {
        Assumptions.assumeTrue(MpsGraphSession.available(),
            "MpsGraph unavailable: ${MpsGraphSession.loadError}")
    }

    @Test
    fun clippingReducesWeightDeltaOverMultipleSteps() {
        ensure()
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
        val rng = Random(101L)
        // 큰 weight로 큰 gradient를 유도
        for (p in tm.parameters()) for (i in 0 until p.numel) p.data[i] = (rng.nextFloat() - 0.5f) * 2.0f

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

        val cfg = MpsGraphConfig(numLayers, embedDim, numHeads, T, vocab, batchSize = 1)
        val firstWShape = tm.parameters()[0].shape
        val numel0 = firstWShape.fold(1) { a, b -> a * b }
        val w0 = tm.parameters()[0].data.copyOf()

        val steps = 5
        val wA = FloatArray(numel0)
        MpsGraphSession.create(cfg).use { s ->
            for ((idx, p) in tm.parameters().withIndex()) s.loadWeights(idx, p.data, p.shape)
            for (step in 1..steps) {
                s.runTrainingStep(tokens, targets, cos, sin, mask,
                    lr = 1e-2f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f,
                    weightDecay = 0.0f, stepT = step, gradClip = 0.0f)
            }
            s.readWeight(0, wA)
        }

        // 매우 작은 clip: m/v 누적량 감소 → 후속 step의 update 점진적 감소
        val wB = FloatArray(numel0)
        MpsGraphSession.create(cfg).use { s ->
            for ((idx, p) in tm.parameters().withIndex()) s.loadWeights(idx, p.data, p.shape)
            for (step in 1..steps) {
                s.runTrainingStep(tokens, targets, cos, sin, mask,
                    lr = 1e-2f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f,
                    weightDecay = 0.0f, stepT = step, gradClip = 1e-5f)
            }
            s.readWeight(0, wB)
        }

        var l2A = 0.0
        var l2B = 0.0
        for (i in 0 until numel0) {
            val dA = (wA[i] - w0[i]).toDouble()
            val dB = (wB[i] - w0[i]).toDouble()
            l2A += dA * dA
            l2B += dB * dB
        }
        l2A = sqrt(l2A)
        l2B = sqrt(l2B)
        assertTrue(l2A > 0.0, "case A 변화량이 0")
        assertTrue(l2B < l2A * 0.8,
            "clipping이 multi-step에서 효과 없음: |ΔA|=$l2A vs |ΔB|=$l2B (< 0.8×여야 정상)")
    }

    @Test
    fun clippingProducesNoNanOrInf() {
        ensure()
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
        val rng = Random(99L)
        for (p in tm.parameters()) for (i in 0 until p.numel) p.data[i] = (rng.nextFloat() - 0.5f) * 0.5f

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

        val cfg = MpsGraphConfig(numLayers, embedDim, numHeads, T, vocab, batchSize = 1)
        MpsGraphSession.create(cfg).use { s ->
            for ((idx, p) in tm.parameters().withIndex()) s.loadWeights(idx, p.data, p.shape)
            for (step in 1..3) {
                val l = s.runTrainingStep(tokens, targets, cos, sin, mask,
                    lr = 1e-2f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f,
                    weightDecay = 0.0f, stepT = step, gradClip = 1e-6f)
                assertTrue(l.isFinite() && !l.isNaN(), "step $step loss=$l invalid")
            }
        }
    }

    @Test
    fun clipDisabledMatchesUnclipped() {
        ensure()
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
        val rng = Random(202L)
        for (p in tm.parameters()) for (i in 0 until p.numel) p.data[i] = (rng.nextFloat() - 0.5f) * 0.1f

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

        val cfg = MpsGraphConfig(numLayers, embedDim, numHeads, T, vocab, batchSize = 1)
        // gradClip=0 (disabled) vs gradClip=1e10 (사실상 disabled)이 같은 결과를 내야.
        val loss0: Float
        val loss1: Float
        MpsGraphSession.create(cfg).use { s ->
            for ((idx, p) in tm.parameters().withIndex()) s.loadWeights(idx, p.data, p.shape)
            loss0 = s.runTrainingStep(tokens, targets, cos, sin, mask,
                lr = 1e-2f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f,
                weightDecay = 0.0f, stepT = 1, gradClip = 0.0f)
        }
        MpsGraphSession.create(cfg).use { s ->
            for ((idx, p) in tm.parameters().withIndex()) s.loadWeights(idx, p.data, p.shape)
            loss1 = s.runTrainingStep(tokens, targets, cos, sin, mask,
                lr = 1e-2f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f,
                weightDecay = 0.0f, stepT = 1, gradClip = 1e10f)
        }
        assertTrue(abs(loss0 - loss1) < 1e-4f,
            "clip disabled와 매우 큰 clip(no-op)의 첫 step loss는 같아야: l0=$loss0 l1=$loss1")
    }
}
