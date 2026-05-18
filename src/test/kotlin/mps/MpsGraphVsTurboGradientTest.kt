package mps

import gpt.GPTConfig
import org.junit.jupiter.api.Assumptions
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue
import turbo.TurboModelConfig
import turbo.TurboTensor
import turbo.layer.TurboPikoGPT

/**
 * turbo와 mps backend의 forward+backward를 1:1 수치 비교 (accum=1, B=1).
 *
 * 검증 목표:
 *   1) 같은 weight + 같은 input → 두 backend의 raw backward weight grad 일치
 *   2) mps의 grad reduction이 graph 내 mean over (B,T)가 맞는지 정량 확인
 *
 * 분석 plan의 가설 1 (sum vs mean accum) 정량 검증:
 *   - mps CE는 graph 내 `mean over (B,T)` → 1 micro grad는 token-mean 스케일
 *   - turbo의 backward는 caller가 upstreamGrad를 결정. 같은 token-mean이면 upstreamGrad = 1/T (B=1)
 *   - 둘이 일치하면 backward 알고리즘 자체는 같고, trainer의 accum 평균화만 차이.
 *   - dropout=0 → 가설 4 분리.
 */
class MpsGraphVsTurboGradientTest {

    @Test
    fun rawForwardBackwardMatches() {
        Assumptions.assumeTrue(
            MpsGraphSession.available(),
            "MpsGraph unavailable: ${MpsGraphSession.loadError}"
        )

        val T = 4
        val embedDim = 16
        val numHeads = 2
        val numLayers = 2
        val vocab = 32
        val seed = 41L
        val headDim = embedDim / numHeads
        val half = headDim / 2

        val gptCfg = GPTConfig(
            vocabularySize = vocab,
            embeddingDimension = embedDim,
            numberOfLayers = numLayers,
            numberOfAttentionHeads = numHeads,
            maxSequenceLength = T,
            dropoutProbability = 0.0f,
            useBias = true,
        )
        val modelCfg = TurboModelConfig(
            gpt = gptCfg,
            tieWeights = true,
            mlpActivation = "gelu",
            positionEncoding = "learned",
            normalizationType = "layernorm",
        )
        val tm = TurboPikoGPT(modelCfg)
        val rng = Random(seed)
        for (param in tm.parameters()) {
            for (i in 0 until param.numel) param.data[i] = (rng.nextFloat() - 0.5f) * 0.2f
        }
        tm.setTraining(false)

        val mask = FloatArray(T * T) { idx -> if (idx % T > idx / T) -1e9f else 0f }
        val cos = FloatArray(T * half) { 1f }
        val sin = FloatArray(T * half) { 0f }
        val cfg = MpsGraphConfig(
            numLayers = numLayers,
            embedDim = embedDim,
            numHeads = numHeads,
            blockSize = T,
            vocab = vocab,
            batchSize = 1,
            useRope = false,
            useSwiglu = false,
            useDropout = false,
            dropoutProbability = 0f,
        )

        val rngIO = Random(seed + 100)
        val tokens = IntArray(T) { rngIO.nextInt(vocab) }
        val targets = IntArray(T) { rngIO.nextInt(vocab) }

        val mpsGrads = MpsGraphSession.create(cfg).use { s ->
            for ((idx, p) in tm.parameters().withIndex()) s.loadWeights(idx, p.data, p.shape)
            s.resetGradAccum()
            s.runAccumStep(tokens, targets, cos, sin, mask, batchSize = 1, dropoutMask = null)
            tm.parameters().mapIndexed { idx, p ->
                val arr = FloatArray(p.numel)
                s.readGrad(idx, arr)
                arr
            }
        }

        val logits = tm.forward(tokens)
        require(logits.shape.contentEquals(intArrayOf(T, vocab))) {
            "turbo logits shape=${logits.shape.contentToString()}, expected [$T, $vocab]"
        }
        val sm = FloatArray(T * vocab)
        for (t in 0 until T) {
            var maxV = Float.NEGATIVE_INFINITY
            for (v in 0 until vocab) maxV = max(maxV, logits.data[t * vocab + v])
            var sumExp = 0.0
            for (v in 0 until vocab) sumExp += exp((logits.data[t * vocab + v] - maxV).toDouble())
            for (v in 0 until vocab) {
                sm[t * vocab + v] = (exp((logits.data[t * vocab + v] - maxV).toDouble()) / sumExp).toFloat()
            }
        }
        val gLogits = TurboTensor(intArrayOf(T, vocab))
        val invT = 1.0f / T.toFloat()
        for (t in 0 until T) {
            for (v in 0 until vocab) {
                val onehot = if (v == targets[t]) 1f else 0f
                gLogits.data[t * vocab + v] = (sm[t * vocab + v] - onehot) * invT
            }
        }
        tm.zeroGrad()
        tm.backward(gLogits)

        var maxAbs = 0.0f
        var maxRel = 0.0f
        var totalCompared = 0
        for ((idx, p) in tm.parameters().withIndex()) {
            val tGrad = p.gradOrAlloc()
            val mGrad = mpsGrads[idx]
            require(tGrad.size == mGrad.size) {
                "param $idx: turbo size=${tGrad.size}, mps size=${mGrad.size}"
            }
            for (i in tGrad.indices) {
                val diff = abs(tGrad[i] - mGrad[i])
                val denom = max(1e-6f, max(abs(tGrad[i]), abs(mGrad[i])))
                val rel = diff / denom
                if (diff > maxAbs) maxAbs = diff
                if (rel > maxRel) maxRel = rel
                totalCompared++
            }
        }
        println("[MpsGraphVsTurboGradientTest] params=${tm.parameters().size} compared=$totalCompared " +
            "maxAbsDiff=$maxAbs maxRelDiff=$maxRel")
        assertTrue(
            maxAbs < 1e-3f && maxRel < 5e-3f,
            "turbo vs mps grad 불일치: maxAbsDiff=$maxAbs, maxRelDiff=$maxRel " +
            "(분석 plan의 가설 1: turbo upstreamGrad=1/T로 token-mean을 맞췄을 때 두 backward가 일치해야 함)"
        )
    }
}
