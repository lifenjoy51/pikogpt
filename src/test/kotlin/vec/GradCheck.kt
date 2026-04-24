package vec

import kotlin.math.abs
import kotlin.test.assertTrue

/**
 * 유한차분(central difference)으로 스칼라 출력 함수 `f(x: Tensor): Float`의
 * 파라미터 `x`에 대한 수치 기울기를 계산한다.
 *
 *   ∂f/∂x_i ≈ ( f(x + ε·e_i) - f(x - ε·e_i) ) / (2ε)
 *
 * 각 원소 하나씩 두 번 호출하므로 O(numel) 계산. 단위 테스트용이라 성능은 무관.
 *
 * 반환값은 `x.data`와 같은 크기의 `FloatArray` 기울기.
 */
fun numericalGradient(
    input: Tensor,
    epsilon: Float = 1e-3f,
    scalarFn: (Tensor) -> Float,
): FloatArray {
    val grad = FloatArray(input.numel)
    for (i in 0 until input.numel) {
        val original = input.data[i]
        input.data[i] = original + epsilon
        val plus = scalarFn(input)
        input.data[i] = original - epsilon
        val minus = scalarFn(input)
        input.data[i] = original  // 복원
        grad[i] = (plus - minus) / (2 * epsilon)
    }
    return grad
}

/**
 * 두 FloatArray가 요소별 허용 오차 내인지 검증.
 * 절대 오차와 상대 오차 모두 고려한 흔한 isclose 형태.
 */
fun assertClose(
    actual: FloatArray,
    expected: FloatArray,
    absTol: Float = 1e-3f,
    relTol: Float = 1e-2f,
    message: String = "",
) {
    assertTrue(actual.size == expected.size, "크기 다름: ${actual.size} vs ${expected.size} $message")
    for (i in actual.indices) {
        val diff = abs(actual[i] - expected[i])
        val limit = absTol + relTol * abs(expected[i])
        assertTrue(
            diff <= limit,
            "[$i] actual=${actual[i]} expected=${expected[i]} diff=$diff > limit=$limit $message"
        )
    }
}
