package turbo.ops

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import turbo.TurboSimdMath
import turbo.TurboTensor
import kotlin.math.sqrt

/**
 * Root Mean Square Normalization (Llama 표준). Phase B에서 SIMD 적용.
 *
 *   Forward:  rms_i  = √(mean(x²) + eps),  y_ij = (x_ij / rms_i) * γ_j
 *   Backward: a_j = gy_j*γ_j, b = mean(x_j*a_j)
 *             dL/dx_k = c * a_k - c³ * b * x_k   where c = 1/rms
 *             dL/dγ_j += Σ_i gy_ij * x_ij * c_i
 */
class TurboRMSNormCache(
    val xData: FloatArray,
    val invRms: FloatArray,
)

fun turboRmsNormForward(
    x: TurboTensor,
    gamma: TurboTensor,
    eps: Float = 1e-5f,
): Pair<TurboTensor, TurboRMSNormCache> {
    require(x.shape.size == 2)
    val n = x.rows
    val c = x.cols
    require(gamma.numel == c) { "γ 차원은 마지막 축과 일치해야 함" }

    val y = TurboTensor(intArrayOf(n, c))
    val invRms = FloatArray(n)
    val invC = 1.0f / c
    val species = TurboSimdMath.SPECIES
    val laneLen = species.length()
    val cUpper = species.loopBound(c)
    val xData = x.data
    val yData = y.data
    val gammaData = gamma.data

    for (i in 0 until n) {
        val rowOff = i * c
        // sumSq SIMD reduction
        var sqAcc = FloatVector.zero(species)
        var j = 0
        while (j < cUpper) {
            val vX = FloatVector.fromArray(species, xData, rowOff + j)
            sqAcc = vX.fma(vX, sqAcc)
            j += laneLen
        }
        var sumSq = sqAcc.reduceLanes(VectorOperators.ADD)
        while (j < c) {
            val v = xData[rowOff + j]
            sumSq += v * v
            j++
        }
        val inv = 1.0f / sqrt(sumSq * invC + eps)
        invRms[i] = inv

        // y = x * inv * γ — d 차원 SIMD
        val vInv = FloatVector.broadcast(species, inv)
        j = 0
        while (j < cUpper) {
            val vX = FloatVector.fromArray(species, xData, rowOff + j)
            val vG = FloatVector.fromArray(species, gammaData, j)
            vX.mul(vInv).mul(vG).intoArray(yData, rowOff + j)
            j += laneLen
        }
        while (j < c) {
            yData[rowOff + j] = xData[rowOff + j] * inv * gammaData[j]
            j++
        }
    }
    return y to TurboRMSNormCache(x.data.copyOf(), invRms)
}

fun turboRmsNormBackward(
    cache: TurboRMSNormCache,
    gamma: TurboTensor,
    gyData: FloatArray,
    rows: Int,
    cols: Int,
): FloatArray {
    val dGamma = gamma.gradOrAlloc()
    val dx = FloatArray(rows * cols)
    val invC = 1.0f / cols
    val species = TurboSimdMath.SPECIES
    val laneLen = species.length()
    val cUpper = species.loopBound(cols)
    val gammaData = gamma.data
    val xData = cache.xData

    for (i in 0 until rows) {
        val rowOff = i * cols
        val invRms = cache.invRms[i]
        val invRms3 = invRms * invRms * invRms
        val vInvRms = FloatVector.broadcast(species, invRms)

        // dGamma[j] += gy*x*invRms — j 차원 SIMD scaled add
        // 동시에 meanXa = Σ x*gy*γ — SIMD reduction
        var meanAcc = FloatVector.zero(species)
        var j = 0
        while (j < cUpper) {
            val vGy = FloatVector.fromArray(species, gyData, rowOff + j)
            val vX = FloatVector.fromArray(species, xData, rowOff + j)
            val vG = FloatVector.fromArray(species, gammaData, j)
            // dGamma += gy * x * invRms
            val vGxC = vGy.mul(vX).mul(vInvRms)
            val vDg = FloatVector.fromArray(species, dGamma, j)
            vGxC.add(vDg).intoArray(dGamma, j)
            // meanXa += x * (gy * γ)
            val vA = vGy.mul(vG)
            meanAcc = vX.fma(vA, meanAcc)
            j += laneLen
        }
        var meanXa = meanAcc.reduceLanes(VectorOperators.ADD)
        while (j < cols) {
            val aj = gyData[rowOff + j] * gammaData[j]
            meanXa += xData[rowOff + j] * aj
            dGamma[j] += gyData[rowOff + j] * xData[rowOff + j] * invRms
            j++
        }
        meanXa *= invC

        // dx = invRms * a - invRms3 * meanXa * x
        val vScale3 = FloatVector.broadcast(species, invRms3 * meanXa)
        j = 0
        while (j < cUpper) {
            val vGy = FloatVector.fromArray(species, gyData, rowOff + j)
            val vG = FloatVector.fromArray(species, gammaData, j)
            val vX = FloatVector.fromArray(species, xData, rowOff + j)
            val vA = vGy.mul(vG)
            // dx = vA*invRms - vX*invRms3*meanXa
            vA.mul(vInvRms).sub(vX.mul(vScale3)).intoArray(dx, rowOff + j)
            j += laneLen
        }
        while (j < cols) {
            val aj = gyData[rowOff + j] * gammaData[j]
            dx[rowOff + j] = invRms * aj - invRms3 * meanXa * xData[rowOff + j]
            j++
        }
    }
    return dx
}
