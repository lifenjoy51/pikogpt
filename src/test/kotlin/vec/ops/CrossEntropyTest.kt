package vec.ops

import vec.Tensor
import vec.assertClose
import vec.numericalGradient
import vec.tensorGaussian
import kotlin.math.ln
import kotlin.test.Test
import kotlin.test.assertTrue

class CrossEntropyTest {

    @Test
    fun uniformLogitsHasBaselineLoss() {
        // 모든 logit 동일 → softmax uniform → loss = ln(V)
        val vocab = 7
        val logits = Tensor(intArrayOf(3, vocab), FloatArray(3 * vocab) { 0.0f })
        val targets = intArrayOf(2, 5, 0)
        val result = crossEntropyForward(logits, targets)
        val expected = ln(vocab.toDouble()).toFloat()
        assertTrue(
            kotlin.math.abs(result.loss - expected) < 1e-4f,
            "uniform loss는 ln($vocab) = $expected 여야 함: got ${result.loss}"
        )
    }

    @Test
    fun crossEntropyBackwardMatchesNumerical() {
        val n = 4
        val vocab = 6
        val logits = tensorGaussian(intArrayOf(n, vocab), std = 0.5f)
        val targets = intArrayOf(1, 3, 0, 5)

        // 수치 기울기
        val numericDLogits = numericalGradient(logits) { ll ->
            crossEntropyForward(ll, targets).loss
        }

        // 분석 backward (loss는 이미 스칼라, upstream grad = 1)
        val res = crossEntropyForward(logits, targets)
        crossEntropyBackward(logits, targets, res.softmax)

        assertClose(logits.grad!!, numericDLogits, message = "cross-entropy backward")
    }
}
