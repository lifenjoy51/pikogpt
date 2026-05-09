package turbo

import gpt.GPTConfig
import turbo.layer.TurboPikoGPT
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Phase 3.0 — KV cache 추론 동등성.
 *
 * 같은 모델 + 같은 토큰 시퀀스에서:
 *   1) full naive forward의 마지막 위치 logits
 *   2) 토큰별 forwardIncremental(cache)의 마지막 logits
 * 가 수치적으로 일치해야 한다 (maxAbsDiff < 1e-4).
 */
class TurboKVCacheTest {

    private fun makeModel(useRoPE: Boolean): TurboPikoGPT {
        val gptCfg = GPTConfig(
            maxSequenceLength = 16,
            vocabularySize = 12,
            numberOfLayers = 2,
            numberOfAttentionHeads = 2,
            embeddingDimension = 8,
            useBias = true,
            dropoutProbability = 0.0f,
        )
        val model = TurboPikoGPT(
            TurboModelConfig(
                gpt = gptCfg,
                positionEncoding = if (useRoPE) "rope" else "learned",
            )
        )
        model.setTraining(false)
        return model
    }

    private fun runComparison(useRoPE: Boolean) {
        val model = makeModel(useRoPE)
        val tokens = intArrayOf(0, 3, 5, 1, 4, 7, 2, 6)

        // 1) naive full-sequence forward
        val naiveLogits = model.forward(tokens)
        val v = naiveLogits.cols
        val tLast = tokens.size - 1
        val naiveLastRow = FloatArray(v) { naiveLogits.data[tLast * v + it] }

        // 2) incremental forward with KV cache
        val gptCfg = model.config.gpt
        val headDim = gptCfg.embeddingDimension / gptCfg.numberOfAttentionHeads
        val kvDim = model.config.effectiveKvHeads * headDim
        val cache = TurboKVCache(
            maxSeqLen = gptCfg.maxSequenceLength,
            numLayers = gptCfg.numberOfLayers,
            kvDim = kvDim,
        )
        var lastLogits: TurboTensor? = null
        for (token in tokens) {
            lastLogits = model.forwardIncremental(token, cache)
        }
        val cacheRow = lastLogits!!.data

        var maxD = 0.0f
        for (i in 0 until v) {
            val d = abs(naiveLastRow[i] - cacheRow[i])
            if (d > maxD) maxD = d
        }
        // 디버그: 첫 4개 비교 출력
        val sb = StringBuilder("[useRoPE=$useRoPE] maxDiff=$maxD\n")
        for (i in 0 until minOf(v, 4)) {
            sb.append("  i=$i naive=${naiveLastRow[i]}  cache=${cacheRow[i]}  diff=${abs(naiveLastRow[i] - cacheRow[i])}\n")
        }
        assertTrue(maxD < 1e-4f, sb.toString())
    }

    @Test
    fun matchesNaiveLearnedPosition() = runComparison(useRoPE = false)

    @Test
    fun matchesNaiveRope() = runComparison(useRoPE = true)
}
