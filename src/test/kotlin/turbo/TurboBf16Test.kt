package turbo

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Phase 4.0 — bf16 변환 정확도.
 *
 *   - bf16는 fp32 상위 16-bit (1 sign + 8 exponent + 7 mantissa)
 *   - mantissa truncate → relative error ≤ 2^-7 ≈ 0.78%
 *   - exponent 범위는 fp32와 동일 → overflow/underflow 새로 발생 안 함
 */
class TurboBf16Test {

    @Test
    fun roundTripPreservesValueWithinPercent() {
        val rng = Random(1)
        val src = FloatArray(1024) { rng.nextFloat() * 200f - 100f }
        val avgRelError = TurboBf16.roundTripAverageRelError(src)
        // 7-bit mantissa truncate → 평균 ~0.4%
        assertTrue(avgRelError < 0.01, "avg relative error too high: $avgRelError")
    }

    @Test
    fun packUnpackArrayMatchesScalarRoundTrip() {
        val rng = Random(2)
        val n = 256
        val src = FloatArray(n) { rng.nextFloat() * 10f - 5f }
        val packed = ShortArray(n)
        TurboBf16.packArray(src, packed)
        val recovered = FloatArray(n)
        TurboBf16.unpackArray(packed, recovered)

        for (i in 0 until n) {
            val expected = TurboBf16.bf16ToFloat(TurboBf16.floatToBf16(src[i]))
            assertTrue(recovered[i] == expected, "mismatch at $i: ${recovered[i]} vs $expected")
        }
    }

    @Test
    fun zeroAndSpecialValues() {
        // 0.0
        assertTrue(TurboBf16.bf16ToFloat(TurboBf16.floatToBf16(0.0f)) == 0.0f)
        // -0.0 (sign bit)
        val negZero = TurboBf16.bf16ToFloat(TurboBf16.floatToBf16(-0.0f))
        assertTrue(negZero == 0.0f || negZero == -0.0f)
        // 1.0 — exponent 범위 안, 정확히 표현
        assertTrue(TurboBf16.bf16ToFloat(TurboBf16.floatToBf16(1.0f)) == 1.0f)
        // 매우 큰 값 (exponent 범위 안에 있어야)
        val big = 1e30f
        val bigBack = TurboBf16.bf16ToFloat(TurboBf16.floatToBf16(big))
        assertTrue(abs((bigBack - big) / big) < 0.01, "big value rel error too high")
    }
}
