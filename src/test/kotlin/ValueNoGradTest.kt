import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ValueNoGradTest {

    @Test
    fun noGradBlockSkipsGraphConstruction() {
        val a = Value(2.0f)
        val b = Value(3.0f)

        val c = GradContext.noGrad { a * b + b.pow(2.0f) }

        // 값은 동일해야 함
        assertEquals(2.0f * 3.0f + 3.0f * 3.0f, c.scalarValue, "no-grad에서도 스칼라 값은 정확해야 함")

        // backward를 호출해도 a/b 그래디언트가 변하지 않아야 함 (그래프가 없으므로)
        a.gradient = 0.0f
        b.gradient = 0.0f
        c.backward()
        assertEquals(0.0f, a.gradient, "no-grad에서 만든 c를 backward 해도 a에 그래디언트 안 붙어야 함")
        assertEquals(0.0f, b.gradient, "no-grad에서 만든 c를 backward 해도 b에 그래디언트 안 붙어야 함")
    }

    @Test
    fun gradModeRestoresAfterNoGrad() {
        val a = Value(2.0f)
        val b = Value(3.0f)

        // 1) no-grad 블록
        GradContext.noGrad {
            a * b
            assertTrue(!GradContext.enabled, "블록 안에서 enabled=false")
        }
        assertTrue(GradContext.enabled, "블록을 벗어나면 enabled=true로 복원")

        // 2) 바깥에서 만든 그래프는 정상 backward 됨
        val c = a * b
        a.gradient = 0.0f
        b.gradient = 0.0f
        c.backward()
        assertEquals(3.0f, a.gradient, "정상 경로에서 d(a*b)/da = b = 3")
        assertEquals(2.0f, b.gradient, "정상 경로에서 d(a*b)/db = a = 2")
    }

    @Test
    fun noGradRestoresEvenOnException() {
        assertTrue(GradContext.enabled, "시작은 enabled=true")
        try {
            GradContext.noGrad {
                throw IllegalStateException("boom")
            }
        } catch (_: IllegalStateException) {
            // 무시
        }
        assertTrue(GradContext.enabled, "예외 시에도 try/finally로 복원")
    }

    @Test
    fun nestedNoGradPreservesOuterState() {
        // 바깥: grad on
        assertTrue(GradContext.enabled)

        GradContext.noGrad {
            assertTrue(!GradContext.enabled)

            GradContext.noGrad {
                // 이미 꺼져 있음; 중첩 호출이 바깥을 깨면 안 됨
                assertTrue(!GradContext.enabled)
            }

            // 중첩 종료 후에도 여전히 꺼져 있어야 함
            assertTrue(!GradContext.enabled)
        }

        // 최상위 블록 종료 후 복원
        assertTrue(GradContext.enabled)
    }
}
