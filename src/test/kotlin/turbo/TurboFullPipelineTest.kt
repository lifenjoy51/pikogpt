package turbo

import gpt.GPTConfig
import turbo.TurboModelConfig
import turbo.layer.TurboPikoGPT
import turbo.ops.turboCrossEntropyBackward
import turbo.ops.turboCrossEntropyForward
import vec.layer.VecPikoGPT
import vec.ops.crossEntropyBackward as vecCEBackward
import vec.ops.crossEntropyForward as vecCEForward
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Phase 0 end-to-end 동등성: 같은 GPTConfig + 같은 RandomGaussian 시퀀스로 두 모델을
 * 만들면 forward(logits)와 backward(weight grads)가 vec과 turbo에서 수치 일치해야 한다.
 *
 * RandomGaussian은 process-wide singleton이므로 동일한 호출 순서로 두 모델을 초기화하면
 * 정확히 같은 weight를 갖는다 (turbo가 같은 layer 생성 순서를 보장).
 *
 * 1초 룰을 위해 매우 작은 모델 사용.
 */
class TurboFullPipelineTest {

    @Test
    fun forwardAndBackwardMatchVec() {
        val gptConfig = GPTConfig(
            maxSequenceLength = 8,
            vocabularySize = 12,
            numberOfLayers = 2,
            numberOfAttentionHeads = 2,
            embeddingDimension = 8,
            useBias = true,
            dropoutProbability = 0.0f,
        )
        // Phase 1 회귀: 모든 turbo 옵션 default OFF — Phase 0 (vec 동등) 경로 보장.
        val turboConfig = TurboModelConfig(gpt = gptConfig)
        check(turboConfig.isPhase0Compatible) { "Phase 0 호환 옵션 조합이어야 회귀 검증 가능" }

        val tokenIds = intArrayOf(0, 3, 5, 1, 4, 7, 2, 6)
        val targets = intArrayOf(3, 5, 1, 4, 7, 2, 6, 0)

        // RandomGaussian은 reset API 없는 process-wide singleton이라 두 모델 가중치가
        // 다르게 초기화될 수 있다. 따라서 vec 모델을 만들고 그 가중치를 turbo로 복사해
        // 동일 시작점을 보장한다.
        val vecModel = VecPikoGPT(gptConfig)
        val turboModel = TurboPikoGPT(turboConfig)
        copyWeightsTurboFromVec(turboModel, vecModel)

        // 양쪽 forward
        val tLogits = turboModel.forward(tokenIds)
        val vLogits = vecModel.forward(tokenIds)

        assertCloseFloatArray(tLogits.data, vLogits.data, 1e-4f, "logits")

        // backward
        val tCe = turboCrossEntropyForward(tLogits, targets)
        val vCe = vecCEForward(vLogits, targets)
        assertTrue(abs(tCe.loss - vCe.loss) < 1e-5f, "loss mismatch ${tCe.loss} vs ${vCe.loss}")

        val tGLogits = turboCrossEntropyBackward(tLogits, targets, tCe.softmax, 1.0f)
        val vGLogits = vecCEBackward(vLogits, targets, vCe.softmax, 1.0f)
        assertCloseFloatArray(tGLogits.data, vGLogits.data, 1e-5f, "grad logits")

        turboModel.backward(tGLogits)
        vecModel.backward(vGLogits)

        // 모든 파라미터 grad 비교
        val tParams = turboModel.parameters()
        val vParams = vecModel.parameters()
        require(tParams.size == vParams.size) { "param count mismatch" }
        for (i in tParams.indices) {
            val tg = tParams[i].grad
            val vg = vParams[i].grad
            require((tg == null) == (vg == null)) { "grad presence mismatch at $i" }
            if (tg != null && vg != null) {
                assertCloseFloatArray(tg, vg, 1e-3f, "param[$i] grad")
            }
        }
    }

    /** turbo 모델의 모든 파라미터 데이터를 vec 모델의 같은 인덱스 파라미터로 덮어씀. */
    private fun copyWeightsTurboFromVec(turbo: TurboPikoGPT, vec: VecPikoGPT) {
        val tParams = turbo.parameters()
        val vParams = vec.parameters()
        require(tParams.size == vParams.size) {
            "param 텐서 수 불일치: turbo=${tParams.size}, vec=${vParams.size}"
        }
        for (i in tParams.indices) {
            require(tParams[i].numel == vParams[i].numel) {
                "param $i numel 불일치: turbo=${tParams[i].numel}, vec=${vParams[i].numel}"
            }
            vParams[i].data.copyInto(tParams[i].data)
        }
    }

    private fun assertCloseFloatArray(a: FloatArray, b: FloatArray, tol: Float, label: String) {
        require(a.size == b.size) { "$label size mismatch: ${a.size} vs ${b.size}" }
        var maxD = 0.0f
        for (i in a.indices) {
            val d = abs(a[i] - b[i])
            if (d > maxD) maxD = d
        }
        assertTrue(maxD <= tol, "$label maxDiff=$maxD > tol=$tol")
    }
}
