package turbo

/**
 * bf16 (Brain Floating Point) 변환 helper — Phase 4의 mixed precision 기반.
 *
 * bf16는 fp32의 상위 16-bit (1 sign + 8 exponent + 7 mantissa). fp32와 같은 exponent 범위라서
 * scaling 없이 직접 변환 가능. mantissa는 fp32 23-bit → bf16 7-bit로 truncate.
 *
 *   fp32 → bf16: 상위 16-bit 추출 (truncate, 또는 round-to-nearest)
 *   bf16 → fp32: zero-extend (하위 16-bit = 0)
 *
 * 메모리: fp32 대비 50% (4 bytes → 2 bytes per scalar)
 * 정확도: 7-bit mantissa는 ~1% relative error. 학습엔 충분 (gradient noise보다 작음).
 *
 * 정책:
 *   - master weight: fp32 (영구)
 *   - forward weight/activation: bf16 (메모리 절감)
 *   - backward grad: fp32 (정확도)
 *   - optimizer state (m, v): fp32
 *
 * Phase 4.0은 변환 함수만. Phase 4.1에서 SIMD bf16 MatMul, 4.2에서 trainer 통합.
 */
object TurboBf16 {

    /** fp32 → bf16 (truncate, 빠름. round-to-nearest는 bias에 약간 차이). */
    @JvmStatic
    fun floatToBf16(f: Float): Short = (f.toRawBits() ushr 16).toShort()

    /** bf16 → fp32 (zero-extend 하위 16-bit). */
    @JvmStatic
    fun bf16ToFloat(b: Short): Float = Float.fromBits(b.toInt() shl 16)

    /** fp32 array → bf16 short array (in-place destination). */
    fun packArray(src: FloatArray, dst: ShortArray) {
        require(src.size == dst.size) { "src/dst size mismatch: ${src.size} vs ${dst.size}" }
        for (i in src.indices) dst[i] = floatToBf16(src[i])
    }

    /** bf16 short array → fp32 array (in-place destination). */
    fun unpackArray(src: ShortArray, dst: FloatArray) {
        require(src.size == dst.size) { "src/dst size mismatch: ${src.size} vs ${dst.size}" }
        for (i in src.indices) dst[i] = bf16ToFloat(src[i])
    }

    /** Round-trip 후 절대 오차 평균 추정 — bf16 mantissa 7-bit 절단 효과 측정. */
    fun roundTripAverageRelError(src: FloatArray): Double {
        var totalRel = 0.0
        var count = 0
        for (i in src.indices) {
            val orig = src[i]
            if (orig == 0.0f) continue
            val recovered = bf16ToFloat(floatToBf16(orig))
            totalRel += kotlin.math.abs((recovered - orig).toDouble() / orig)
            count++
        }
        return if (count == 0) 0.0 else totalRel / count
    }

    // ---- Phase 4.1: weight bf16 storage helper ----

    /** TurboTensor를 bf16 packed ShortArray로 변환 (메모리 50% 절감, 정확도 ~0.4% 손실). */
    fun packTensor(src: TurboTensor): ShortArray {
        val out = ShortArray(src.numel)
        packArray(src.data, out)
        return out
    }

    /** bf16 packed ShortArray + shape → fp32 TurboTensor 복원. forward 직전에 unpack해 사용. */
    fun unpackToTensor(packed: ShortArray, shape: IntArray): TurboTensor {
        val data = FloatArray(packed.size)
        unpackArray(packed, data)
        return TurboTensor(shape, data)
    }
}
