package turbo

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorSpecies

/**
 * Java Vector API SIMD 헬퍼 — Phase 2의 hot path 가속에 사용.
 *
 * 플랫폼별 lane 폭:
 *   - Apple Silicon NEON: 4 lanes (128-bit)
 *   - x86 AVX2: 8 lanes (256-bit)
 *   - x86 AVX-512: 16 lanes (512-bit)
 *
 * `SPECIES_PREFERRED`로 추상화 — 코드 한 번 작성, 모든 플랫폼에서 최대 lane 활용.
 */
object TurboSimdMath {
    @JvmField
    val SPECIES: VectorSpecies<Float> = FloatVector.SPECIES_PREFERRED

    /** 현재 플랫폼의 SIMD lane 수 (디버깅/벤치 보고용). */
    val laneCount: Int get() = SPECIES.length()

    /**
     * c[i] += a[i] * scalar  (FMA 1-pass, scalar가 상수일 때 hot loop에 적합).
     */
    fun fmaScalar(a: FloatArray, scalar: Float, c: FloatArray, len: Int) {
        val species = SPECIES
        val upper = species.loopBound(len)
        val vScalar = FloatVector.broadcast(species, scalar)
        var i = 0
        while (i < upper) {
            val va = FloatVector.fromArray(species, a, i)
            val vc = FloatVector.fromArray(species, c, i)
            va.fma(vScalar, vc).intoArray(c, i)
            i += species.length()
        }
        while (i < len) {
            c[i] += a[i] * scalar
            i++
        }
    }

    /** out[i] = a[i] + b[i]. */
    fun addArrays(a: FloatArray, b: FloatArray, out: FloatArray, len: Int) {
        val species = SPECIES
        val upper = species.loopBound(len)
        var i = 0
        while (i < upper) {
            val va = FloatVector.fromArray(species, a, i)
            val vb = FloatVector.fromArray(species, b, i)
            va.add(vb).intoArray(out, i)
            i += species.length()
        }
        while (i < len) { out[i] = a[i] + b[i]; i++ }
    }

    /** dot product Σ a[i] * b[i] over [0, len). */
    fun dot(a: FloatArray, aOffset: Int, b: FloatArray, bOffset: Int, len: Int): Float {
        val species = SPECIES
        val upper = species.loopBound(len)
        var acc = FloatVector.zero(species)
        var i = 0
        while (i < upper) {
            val va = FloatVector.fromArray(species, a, aOffset + i)
            val vb = FloatVector.fromArray(species, b, bOffset + i)
            acc = va.fma(vb, acc)
            i += species.length()
        }
        var sum = acc.reduceLanes(jdk.incubator.vector.VectorOperators.ADD)
        while (i < len) {
            sum += a[aOffset + i] * b[bOffset + i]
            i++
        }
        return sum
    }
}
