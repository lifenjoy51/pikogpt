package turbo.ops

import jdk.incubator.vector.FloatVector
import turbo.TurboSimdMath
import turbo.TurboTensor

/**
 * 2차원 행렬 곱. Phase 2부터 Java Vector API로 inner loop SIMD 가속.
 *
 *   Shape:     A[M, K] · B[K, N] → C[M, N]
 *   Forward:   "ikj" 루프 — 가장 안쪽 j 차원을 SIMD lane으로 처리.
 *              C[i, :] += A[i, k] * B[k, :]   (lane-wise FMA)
 *   Backward:  ∂L/∂A[i, kk] = Σ_j gy[i, j] * B[kk, j]    ← j축 SIMD dot
 *              ∂L/∂B[kk, j] += A[i, kk] * gy[i, j]      ← j축 SIMD broadcast
 *
 * Phase 0~1과 수치적으로 동일 (float32 누적 순서가 같은 ikj). SIMD lane 처리 순서는
 * "j를 lane 단위로 묶어 처리"라 동일 누적 순서를 유지 (각 j에 대한 contribution은
 * naive와 동일 a[k] × b[k,j]).
 */
fun turboMatmul(a: TurboTensor, b: TurboTensor): TurboTensor {
    require(a.shape.size == 2 && b.shape.size == 2) {
        "turboMatmul은 2차원 텐서만: a=${a.shape.contentToString()}, b=${b.shape.contentToString()}"
    }
    val m = a.rows
    val k = a.cols
    val n = b.cols
    require(b.rows == k) { "차원 불일치: A[$m, $k] · B[${b.rows}, $n]" }

    val c = TurboTensor(intArrayOf(m, n))
    val species = TurboSimdMath.SPECIES
    val nUpper = species.loopBound(n)
    val cd = c.data
    val ad = a.data
    val bd = b.data

    for (i in 0 until m) {
        val aRowOffset = i * k
        val cRowOffset = i * n
        for (kk in 0 until k) {
            val aik = ad[aRowOffset + kk]
            val bRowOffset = kk * n
            val vScalar = FloatVector.broadcast(species, aik)
            var j = 0
            while (j < nUpper) {
                val vc = FloatVector.fromArray(species, cd, cRowOffset + j)
                val vb = FloatVector.fromArray(species, bd, bRowOffset + j)
                vb.fma(vScalar, vc).intoArray(cd, cRowOffset + j)
                j += species.length()
            }
            while (j < n) {
                cd[cRowOffset + j] += aik * bd[bRowOffset + j]
                j++
            }
        }
    }
    return c
}

/**
 * MatMul 백엔드 dispatch toggle.
 *
 * 기본은 turboMatmul / turboMatmulBackward (CPU SIMD).
 * `mps.MpsBackend.enable()`이 호출되면 Metal GPU 구현으로 교체된다.
 *
 * 함수 참조 방식: TurboLinear/TurboPikoGPT의 시그니처를 바꾸지 않으면서 백엔드 교체 가능.
 */
@Volatile
var matmulImpl: (TurboTensor, TurboTensor) -> TurboTensor =
    { a, b -> turboMatmul(a, b) }

@Volatile
var matmulBackwardImpl: (TurboTensor, TurboTensor, FloatArray) -> Unit =
    { a, b, gy -> turboMatmulBackward(a, b, gy) }

fun turboMatmulBackward(a: TurboTensor, b: TurboTensor, gyData: FloatArray) {
    val m = a.rows
    val k = a.cols
    val n = b.cols
    require(gyData.size == m * n) { "gy 크기 불일치: expected ${m * n}, got ${gyData.size}" }

    val dA = a.gradOrAlloc()
    val dB = b.gradOrAlloc()
    val ad = a.data
    val bd = b.data
    val species = TurboSimdMath.SPECIES
    val nUpper = species.loopBound(n)

    // dA[i, kk] = Σ_j gy[i, j] * B[kk, j]   — j축 SIMD dot
    for (i in 0 until m) {
        val gyOff = i * n
        val dAOff = i * k
        for (kk in 0 until k) {
            val bOff = kk * n
            var acc = FloatVector.zero(species)
            var j = 0
            while (j < nUpper) {
                val vGy = FloatVector.fromArray(species, gyData, gyOff + j)
                val vB = FloatVector.fromArray(species, bd, bOff + j)
                acc = vGy.fma(vB, acc)
                j += species.length()
            }
            var sum = acc.reduceLanes(jdk.incubator.vector.VectorOperators.ADD)
            while (j < n) {
                sum += gyData[gyOff + j] * bd[bOff + j]
                j++
            }
            dA[dAOff + kk] += sum
        }
    }

    // dB[kk, j] += Σ_i A[i, kk] * gy[i, j]  — i 외부, j SIMD broadcast scaled add
    for (i in 0 until m) {
        val aOff = i * k
        val gyOff = i * n
        for (kk in 0 until k) {
            val aik = ad[aOff + kk]
            val dBOff = kk * n
            val vScalar = FloatVector.broadcast(species, aik)
            var j = 0
            while (j < nUpper) {
                val vGy = FloatVector.fromArray(species, gyData, gyOff + j)
                val vDb = FloatVector.fromArray(species, dB, dBOff + j)
                vGy.fma(vScalar, vDb).intoArray(dB, dBOff + j)
                j += species.length()
            }
            while (j < n) {
                dB[dBOff + j] += aik * gyData[gyOff + j]
                j++
            }
        }
    }
}
