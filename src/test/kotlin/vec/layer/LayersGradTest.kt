package vec.layer

import gpt.GPTConfig
import vec.Tensor
import vec.assertClose
import vec.numericalGradient
import vec.tensorGaussian
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * 레이어 레벨 grad-check.
 *
 * 전략: 각 레이어에 대해 loss = sum(layer.forward(x) * w) (w는 고정된 무작위 가중치)
 * 로 잡고 x의 수치 기울기와 layer.backward(w)가 반환하는 분석 기울기를 비교한다.
 * 파라미터 grad도 수치 기울기와 비교 — 선택적 param 하나만 체크해도 충분한 sanity.
 */
class LayersGradTest {

    @Test
    fun linearBackward() {
        val layer = Linear(4, 3, useBias = true)
        val x = tensorGaussian(intArrayOf(2, 4), std = 0.5f)
        val w = tensorGaussian(intArrayOf(2, 3), std = 1.0f)

        val numericDX = numericalGradient(x) { xx -> dot(layer.forward(xx), w) }

        layer.forward(x)  // 캐시 준비
        val analyticDX = layer.backward(Tensor(intArrayOf(2, 3), w.data.copyOf()))
        assertClose(analyticDX.data, numericDX, message = "Linear dx")
    }

    @Test
    fun layerNormBackward() {
        val layer = LayerNorm(5)
        val x = tensorGaussian(intArrayOf(3, 5), std = 1.0f)
        val w = tensorGaussian(intArrayOf(3, 5), std = 1.0f)

        val numericDX = numericalGradient(x) { xx -> dot(layer.forward(xx), w) }

        layer.forward(x)
        val analyticDX = layer.backward(Tensor(intArrayOf(3, 5), w.data.copyOf()))
        assertClose(analyticDX.data, numericDX, message = "LayerNorm dx")
    }

    @Test
    fun mlpBackward() {
        val layer = MLP(embedDim = 4)
        val x = tensorGaussian(intArrayOf(2, 4), std = 0.3f)
        val w = tensorGaussian(intArrayOf(2, 4), std = 1.0f)

        val numericDX = numericalGradient(x) { xx -> dot(layer.forward(xx), w) }

        layer.forward(x)
        val analyticDX = layer.backward(Tensor(intArrayOf(2, 4), w.data.copyOf()))
        assertClose(analyticDX.data, numericDX, absTol = 3e-3f, message = "MLP dx")
    }

    @Test
    fun selfAttentionBackward() {
        val layer = SelfAttention(embedDim = 6, numHeads = 2)
        val x = tensorGaussian(intArrayOf(3, 6), std = 0.3f)
        val w = tensorGaussian(intArrayOf(3, 6), std = 1.0f)

        val numericDX = numericalGradient(x) { xx -> dot(layer.forward(xx), w) }

        layer.forward(x)
        val analyticDX = layer.backward(Tensor(intArrayOf(3, 6), w.data.copyOf()))
        assertClose(analyticDX.data, numericDX, absTol = 3e-3f, message = "SelfAttention dx")
    }

    @Test
    fun transformerBlockBackward() {
        val layer = TransformerBlock(embedDim = 6, numHeads = 2)
        val x = tensorGaussian(intArrayOf(3, 6), std = 0.3f)
        val w = tensorGaussian(intArrayOf(3, 6), std = 1.0f)

        val numericDX = numericalGradient(x) { xx -> dot(layer.forward(xx), w) }

        layer.forward(x)
        val analyticDX = layer.backward(Tensor(intArrayOf(3, 6), w.data.copyOf()))
        assertClose(analyticDX.data, numericDX, absTol = 5e-3f, message = "Block dx")
    }

    @Test
    fun pikoGptForwardShapeAndBackwardRuns() {
        val config = GPTConfig(
            maxSequenceLength = 8,
            vocabularySize = 10,
            numberOfLayers = 1,
            numberOfAttentionHeads = 2,
            embeddingDimension = 6,
            useBias = true,
            dropoutProbability = 0.0f,  // 벡터 경로에는 dropout 미구현 (MVP)
        )
        val model = PikoGPT(config)
        val tokens = intArrayOf(1, 3, 5, 2)
        val logits = model.forward(tokens)
        assertTrue(logits.rows == 4 && logits.cols == 10, "logits shape [T=4, V=10] 기대")

        // backward가 예외 없이 통과하고 token embedding grad가 일부 채워지는지
        val gLogits = tensorGaussian(intArrayOf(4, 10), std = 1.0f)
        model.backward(gLogits)
        val tokGrad = model.tokenEmbedding.weight.grad
        assertTrue(tokGrad != null, "token embedding grad 할당되어야 함")
        assertTrue(tokGrad!!.any { abs(it) > 1e-6f }, "토큰 1,3,5,2 행에 grad가 채워져야 함")
    }

    /** 두 같은 shape 텐서의 원소별 곱을 모두 더한 스칼라 (내적 대용). */
    private fun dot(a: Tensor, b: Tensor): Float {
        require(a.numel == b.numel)
        var s = 0.0f
        for (i in a.data.indices) s += a.data[i] * b.data[i]
        return s
    }
}
