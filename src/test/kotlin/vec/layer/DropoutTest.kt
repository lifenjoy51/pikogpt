package vec.layer

import vec.Tensor
import vec.tensorGaussian
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Dropout 레이어 단위 테스트.
 *
 * 랜덤성이 있어 수치 기울기와의 비교는 곤란 (mask가 매번 새로 뽑힘). 대신:
 *  - p=0이면 identity
 *  - training=false면 identity
 *  - p>0 + training=true일 때 inverted 스케일(E[mask]=1) 확인 (충분히 많은 원소로 평균)
 *  - backward가 forward mask를 재사용하는지 → "forward 결과의 비영 원소 위치에서만 backward 전달" 확인
 */
class DropoutTest {

    @Test
    fun identityWhenProbZero() {
        val d = Dropout(0.0f)
        val x = tensorGaussian(intArrayOf(4, 8), std = 1.0f)
        val y = d.forward(x)
        // 같은 FloatArray를 그대로 쓰는 게 계약 — shape도 동일.
        assertTrue(x.data.contentEquals(y.data), "p=0이면 forward는 identity")
        val gy = tensorGaussian(intArrayOf(4, 8), std = 1.0f)
        val dx = d.backward(gy)
        assertTrue(gy.data.contentEquals(dx.data), "p=0이면 backward도 identity")
    }

    @Test
    fun identityWhenEvalMode() {
        val d = Dropout(0.5f).apply { training = false }
        val x = tensorGaussian(intArrayOf(4, 8), std = 1.0f)
        val y = d.forward(x)
        assertTrue(x.data.contentEquals(y.data), "training=false면 forward는 identity")
        val gy = tensorGaussian(intArrayOf(4, 8), std = 1.0f)
        val dx = d.backward(gy)
        assertTrue(gy.data.contentEquals(dx.data), "training=false면 backward도 identity")
    }

    @Test
    fun trainingModeAppliesInvertedScaling() {
        // x를 전부 1로 두면 y = mask 자체. 평균이 1.0f 근처여야 (inverted dropout 기대값 보존).
        val n = 10000
        val d = Dropout(0.3f).apply { training = true }
        val x = Tensor(intArrayOf(n)).also { it.data.fill(1.0f) }
        val y = d.forward(x)

        // 1) 모든 원소는 0 또는 1/(1-p) 중 하나여야 함.
        val keep = 1.0f / (1.0f - 0.3f)
        for (v in y.data) {
            assertTrue(v == 0.0f || abs(v - keep) < 1e-5f, "dropout 원소는 0 또는 1/(1-p)여야: v=$v")
        }

        // 2) 평균이 1.0f 근처인지 (inverted scaling 정확성).
        val mean = y.data.average().toFloat()
        assertTrue(abs(mean - 1.0f) < 0.05f, "기대값 1.0 ± 0.05, actual=$mean")
    }

    @Test
    fun backwardReusesForwardMask() {
        // 구조 확인: dx[i]는 gy[i] * mask[i] 이고, mask가 0인 위치에서 dx도 0, mask가 1/(1-p)면
        // dx = gy * 1/(1-p) 정확히 성립. 즉 forward y[i]==0 ⇔ backward dx[i]==0.
        val n = 2000
        val d = Dropout(0.5f).apply { training = true }
        val x = Tensor(intArrayOf(n)).also { it.data.fill(1.0f) }
        val y = d.forward(x)

        val gy = Tensor(intArrayOf(n)).also { for (i in 0 until n) it.data[i] = (i + 1).toFloat() }
        val dx = d.backward(gy)

        val keep = 1.0f / (1.0f - 0.5f)
        for (i in 0 until n) {
            if (y.data[i] == 0.0f) {
                assertEquals(0.0f, dx.data[i], "mask=0 위치에서 dx=0 이어야: i=$i")
            } else {
                val expected = gy.data[i] * keep
                assertTrue(
                    abs(dx.data[i] - expected) < 1e-4f,
                    "mask!=0 위치에서 dx=gy*scale 이어야: i=$i, expected=$expected, actual=${dx.data[i]}",
                )
            }
        }
    }

    @Test
    fun differentMasksAcrossForwardCalls() {
        // 두 번째 forward에서는 새 mask가 뽑혀야 함 — 같은 입력이어도 다른 출력.
        val n = 2000
        val d = Dropout(0.5f).apply { training = true }
        val x = Tensor(intArrayOf(n)).also { it.data.fill(1.0f) }
        val y1 = d.forward(x).data.copyOf()
        val y2 = d.forward(x).data.copyOf()

        // 완전히 같을 확률은 (p^n + (1-p)^n)^something — 사실상 0.
        assertTrue(!y1.contentEquals(y2), "forward를 다시 호출하면 새 mask가 뽑혀야 함")
    }
}
