package vec.ops

import vec.Tensor
import vec.assertClose
import vec.numericalGradient
import vec.tensorGaussian
import kotlin.test.Test

/**
 * MatMul의 backward 검증.
 *
 * 전략: 무작위 A, B를 잡고 loss = sum(matmul(A, B))를 정의한 뒤
 *   - 수치 기울기(`numericalGradient`) = 진짜 정답 근사
 *   - 분석 기울기(`matmulBackward(A, B, ones)`) = 구현된 backward
 * 두 값이 허용 오차 안에 있는지 확인한다.
 *
 * 손실이 sum이므로 ∂loss/∂C = ones. 이 값을 gy로 넘기면 된다.
 */
class MatMulTest {

    @Test
    fun matmulBackwardMatchesNumerical() {
        val a = tensorGaussian(intArrayOf(3, 4), std = 0.3f)
        val b = tensorGaussian(intArrayOf(4, 5), std = 0.3f)

        // 분석 backward: loss = sum(C), ∂loss/∂C = 1 everywhere
        val ones = FloatArray(a.rows * b.cols) { 1.0f }
        matmulBackward(a, b, ones)
        val analyticDA = a.grad!!.copyOf()
        val analyticDB = b.grad!!.copyOf()

        // 수치 기울기: A를 흔들어 보면서 loss 변화 관찰
        val numericDA = numericalGradient(a) { x -> matmul(x, b).data.sum() }
        val numericDB = numericalGradient(b) { y -> matmul(a, y).data.sum() }

        assertClose(analyticDA, numericDA, message = "∂loss/∂A")
        assertClose(analyticDB, numericDB, message = "∂loss/∂B")
    }

    @Test
    fun matmulForwardShape() {
        val a = Tensor(intArrayOf(2, 3), floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f))
        val b = Tensor(intArrayOf(3, 2), floatArrayOf(1f, 0f, 0f, 1f, 1f, 1f))
        val c = matmul(a, b)
        // c[0] = [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5]
        // c[1] = [4*1+5*0+6*1, 4*0+5*1+6*1] = [10, 11]
        assertClose(c.data, floatArrayOf(4f, 5f, 10f, 11f))
    }

    @Test
    fun matmulGradientAccumulates() {
        // backward를 두 번 호출하면 grad가 누적되는지
        val a = tensorGaussian(intArrayOf(2, 3))
        val b = tensorGaussian(intArrayOf(3, 2))
        val ones = FloatArray(a.rows * b.cols) { 1.0f }

        matmulBackward(a, b, ones)
        val once = a.grad!!.copyOf()
        matmulBackward(a, b, ones)  // 두 번째 호출, 누적
        val twice = a.grad!!.copyOf()

        // twice == once * 2 (근사)
        for (i in once.indices) {
            val expected = once[i] * 2
            kotlin.test.assertTrue(
                kotlin.math.abs(twice[i] - expected) < 1e-5f,
                "누적 실패: twice=${twice[i]} expected=$expected"
            )
        }
    }
}
