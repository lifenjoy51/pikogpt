package turbo

import gpt.GPTConfig
import turbo.layer.TurboPikoGPT
import turbo.ops.turboCrossEntropyBackward
import turbo.ops.turboCrossEntropyForward
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Phase 1.1 — qk-norm 기본 동작 검증.
 *   - useQkNorm=true 모델 forward 결과는 NaN/Inf 없음
 *   - backward 후 q/k norm γ에 grad 누적
 *   - useQkNorm=false 모델은 q/k norm 파라미터 0개 (회귀 보장)
 */
class TurboQkNormTest {

    private fun makeConfig(useQkNorm: Boolean): TurboModelConfig {
        return TurboModelConfig(
            gpt = GPTConfig(
                maxSequenceLength = 8,
                vocabularySize = 12,
                numberOfLayers = 2,
                numberOfAttentionHeads = 2,
                embeddingDimension = 8,
                useBias = true,
                dropoutProbability = 0.0f,
                positionEncoding = "rope",
            ),
            normalizationType = "rmsnorm",
            useQkNorm = useQkNorm,
        )
    }

    @Test
    fun qkNormDisabledHasNoExtraParams() {
        val baseline = TurboPikoGPT(makeConfig(useQkNorm = false))
        val qkNormed = TurboPikoGPT(makeConfig(useQkNorm = true))

        // 차이는 numberOfLayers × 2 (q,k) × headDim 만큼의 RMSNorm γ.
        val diff = qkNormed.parameters().sumOf { it.numel } - baseline.parameters().sumOf { it.numel }
        // 2 layers × 2 (qNorm/kNorm) × headDim(=4) = 16
        assertEquals(16, diff, "qk-norm 활성화 시 추가 파라미터 수 계산 mismatch")
    }

    @Test
    fun qkNormEnabledForwardBackwardRunsWithoutNaN() {
        val model = TurboPikoGPT(makeConfig(useQkNorm = true))
        val tokenIds = intArrayOf(0, 3, 5, 1, 4, 7, 2, 6)
        val targets = intArrayOf(3, 5, 1, 4, 7, 2, 6, 0)

        val logits = model.forward(tokenIds)
        for (v in logits.data) {
            assertTrue(v.isFinite(), "logits에 NaN/Inf 발생: $v")
        }

        val ce = turboCrossEntropyForward(logits, targets)
        assertTrue(ce.loss.isFinite(), "loss NaN/Inf: ${ce.loss}")

        val gLogits = turboCrossEntropyBackward(logits, targets, ce.softmax, 1.0f)
        model.backward(gLogits)

        // qNorm/kNorm 파라미터에 grad 누적됐는지 확인 (block 0 attention)
        val attn0 = model.blocks[0].attention
        val qNormGrad = attn0.qNorm?.parameters()?.firstOrNull()?.grad
        val kNormGrad = attn0.kNorm?.parameters()?.firstOrNull()?.grad
        assertTrue(qNormGrad != null, "qNorm grad가 alloc 되어야")
        assertTrue(kNormGrad != null, "kNorm grad가 alloc 되어야")
        // grad가 실제로 0이 아닌 값을 가지는지 (단조롭게 0이면 backward 누락 의심)
        var anyNonzeroQ = false
        var anyNonzeroK = false
        for (g in qNormGrad) if (g != 0.0f) { anyNonzeroQ = true; break }
        for (g in kNormGrad) if (g != 0.0f) { anyNonzeroK = true; break }
        assertTrue(anyNonzeroQ, "qNorm grad가 모두 0 — backward 누락 의심")
        assertTrue(anyNonzeroK, "kNorm grad가 모두 0 — backward 누락 의심")
    }
}
