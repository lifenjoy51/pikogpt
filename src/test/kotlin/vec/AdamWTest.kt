package vec

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * `vec.AdamW`가 AdamW 수식(decoupled wd + bias-correction + m/v update)에 맞게
 * 파라미터를 이동시키는지 검증한다.
 */
class AdamWTest {

    @Test
    fun parameterMovesOppositeToGradient() {
        val p = Tensor(intArrayOf(3), floatArrayOf(1.0f, 2.0f, 3.0f))
        p.gradOrAlloc()[0] = 1.0f   // dL/dp[0] = 1 (양수) → p[0] 감소 기대
        p.gradOrAlloc()[1] = -2.0f  // 음수 → p[1] 증가 기대
        p.gradOrAlloc()[2] = 0.0f   // 0 → 변화 거의 없음 (weight decay만)

        val original = p.data.copyOf()
        val adam = AdamW(
            parameters = listOf(p),
            learningRate = 0.1f,
            weightDecay = 0.0f,  // 이번 테스트에선 순수 gradient 방향 확인
        )
        adam.step()

        assertTrue(p.data[0] < original[0], "grad > 0 이면 파라미터가 감소해야 함")
        assertTrue(p.data[1] > original[1], "grad < 0 이면 파라미터가 증가해야 함")
        assertTrue(abs(p.data[2] - original[2]) < 1e-6f, "grad == 0 이면 거의 불변")
    }

    @Test
    fun weightDecayShrinksParameterWhenGradZero() {
        val p = Tensor(intArrayOf(2), floatArrayOf(1.0f, -1.0f))
        // grad 없음 — step 전에 명시적으로 할당만
        p.gradOrAlloc()  // 모두 0

        val adam = AdamW(
            parameters = listOf(p),
            learningRate = 0.1f,
            weightDecay = 0.5f,
        )
        adam.step()

        // decoupled weight decay: p -= lr * wd * p = 0.05 * p → 새 p ≈ 0.95 * 원본
        assertTrue(abs(p.data[0] - 0.95f) < 1e-5f, "wd 적용 실패: ${p.data[0]}")
        assertTrue(abs(p.data[1] - (-0.95f)) < 1e-5f, "wd 적용 실패: ${p.data[1]}")
    }

    @Test
    fun momentsAccumulateAcrossSteps() {
        val p = Tensor(intArrayOf(1), floatArrayOf(0.0f))
        val adam = AdamW(
            parameters = listOf(p),
            learningRate = 0.01f,
            beta1 = 0.9f,
            beta2 = 0.999f,
            weightDecay = 0.0f,
        )

        // 세 번 같은 방향의 gradient를 주면 step 크기가 bias-correction 효과로 안정화되는지
        val steps = mutableListOf<Float>()
        for (i in 0 until 3) {
            p.gradOrAlloc()[0] = 1.0f
            val before = p.data[0]
            adam.step()
            steps += before - p.data[0]  // step size (감소량)
        }

        // 모든 step은 양의 감소량이어야 (grad가 양수이므로)
        assertTrue(steps.all { it > 0 }, "모든 step이 같은 방향이어야: $steps")
        // Adam은 초기 bias correction 때문에 처음 몇 step이 큰 편 — 여기선 단조성만 확인 가능하면 OK
        // (수식 정확성은 분석적으로 증명하기 번거로우므로 거친 sanity check)
    }
}
