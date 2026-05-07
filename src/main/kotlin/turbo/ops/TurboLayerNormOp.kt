package turbo.ops

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import turbo.TurboSimdMath
import turbo.TurboTensor
import kotlin.math.sqrt

/**
 * Row-wise LayerNorm. Phase B에서 SIMD 적용 — reduction (sum/sumSq) + 정규화 inner SIMD.
 *
 *   y = γ · (x - μ)/√(σ² + ε) + β
 */
class TurboLayerNormCache(
    val xHat: FloatArray,
    val invStd: FloatArray,
)

fun turboLayerNormForward(
    x: TurboTensor,
    gamma: TurboTensor,
    beta: TurboTensor,
    eps: Float = 1e-5f,
): Pair<TurboTensor, TurboLayerNormCache> {
    require(x.shape.size == 2)
    val n = x.rows
    val c = x.cols
    require(gamma.numel == c && beta.numel == c) {
        "γ,β 차원은 마지막 축과 일치해야 함"
    }

    val y = TurboTensor(intArrayOf(n, c))
    val xHat = FloatArray(n * c)
    val invStd = FloatArray(n)
    val species = TurboSimdMath.SPECIES
    val laneLen = species.length()
    val cUpper = species.loopBound(c)
    val xData = x.data
    val yData = y.data
    val gammaData = gamma.data
    val betaData = beta.data
    val invC = 1.0f / c

    for (i in 0 until n) {
        val rowOff = i * c
        // 1) mean = Σ x / c (SIMD reduction)
        var meanAcc = FloatVector.zero(species)
        var j = 0
        while (j < cUpper) {
            meanAcc = meanAcc.add(FloatVector.fromArray(species, xData, rowOff + j))
            j += laneLen
        }
        var mean = meanAcc.reduceLanes(VectorOperators.ADD)
        while (j < c) { mean += xData[rowOff + j]; j++ }
        mean *= invC

        // 2) variance = Σ (x-mean)^2 / c (SIMD fma reduction)
        val vMean = FloatVector.broadcast(species, mean)
        var varAcc = FloatVector.zero(species)
        j = 0
        while (j < cUpper) {
            val vX = FloatVector.fromArray(species, xData, rowOff + j)
            val vD = vX.sub(vMean)
            varAcc = vD.fma(vD, varAcc)
            j += laneLen
        }
        var variance = varAcc.reduceLanes(VectorOperators.ADD)
        while (j < c) {
            val d = xData[rowOff + j] - mean
            variance += d * d
            j++
        }
        variance *= invC
        val inv = 1.0f / sqrt(variance + eps)
        invStd[i] = inv

        // 3) y = γ * (x-mean)*inv + β  + xHat 캐시
        val vInv = FloatVector.broadcast(species, inv)
        j = 0
        while (j < cUpper) {
            val vX = FloatVector.fromArray(species, xData, rowOff + j)
            val vH = vX.sub(vMean).mul(vInv)
            vH.intoArray(xHat, rowOff + j)
            val vG = FloatVector.fromArray(species, gammaData, j)
            val vB = FloatVector.fromArray(species, betaData, j)
            vH.fma(vG, vB).intoArray(yData, rowOff + j)
            j += laneLen
        }
        while (j < c) {
            val h = (xData[rowOff + j] - mean) * inv
            xHat[rowOff + j] = h
            yData[rowOff + j] = gammaData[j] * h + betaData[j]
            j++
        }
    }
    return y to TurboLayerNormCache(xHat, invStd)
}

fun turboLayerNormBackward(
    cache: TurboLayerNormCache,
    gamma: TurboTensor,
    beta: TurboTensor,
    gyData: FloatArray,
    rows: Int,
    cols: Int,
): FloatArray {
    val dGamma = gamma.gradOrAlloc()
    val dBeta = beta.gradOrAlloc()
    val dx = FloatArray(rows * cols)
    val species = TurboSimdMath.SPECIES
    val laneLen = species.length()
    val cUpper = species.loopBound(cols)
    val gammaData = gamma.data
    val xHat = cache.xHat
    val invC = 1.0f / cols

    for (i in 0 until rows) {
        val rowOff = i * cols

        // dGamma[j] += gy[i,j] * xHat[i,j]; dBeta[j] += gy[i,j] — j 차원 SIMD scaled add
        var j = 0
        while (j < cUpper) {
            val vGy = FloatVector.fromArray(species, gyData, rowOff + j)
            val vXh = FloatVector.fromArray(species, xHat, rowOff + j)
            val vDg = FloatVector.fromArray(species, dGamma, j)
            vGy.fma(vXh, vDg).intoArray(dGamma, j)
            val vDb = FloatVector.fromArray(species, dBeta, j)
            vGy.add(vDb).intoArray(dBeta, j)
            j += laneLen
        }
        while (j < cols) {
            dGamma[j] += gyData[rowOff + j] * xHat[rowOff + j]
            dBeta[j] += gyData[rowOff + j]
            j++
        }

        // meanDxHat = Σ (gy*γ); meanDxHatXHat = Σ (gy*γ*xHat) — SIMD reduction
        var meanAcc = FloatVector.zero(species)
        var meanXAcc = FloatVector.zero(species)
        j = 0
        while (j < cUpper) {
            val vGy = FloatVector.fromArray(species, gyData, rowOff + j)
            val vG = FloatVector.fromArray(species, gammaData, j)
            val vDxHat = vGy.mul(vG)
            meanAcc = meanAcc.add(vDxHat)
            val vXh = FloatVector.fromArray(species, xHat, rowOff + j)
            meanXAcc = vDxHat.fma(vXh, meanXAcc)
            j += laneLen
        }
        var meanDxHat = meanAcc.reduceLanes(VectorOperators.ADD)
        var meanDxHatXHat = meanXAcc.reduceLanes(VectorOperators.ADD)
        while (j < cols) {
            val dxHat = gyData[rowOff + j] * gammaData[j]
            meanDxHat += dxHat
            meanDxHatXHat += dxHat * xHat[rowOff + j]
            j++
        }
        meanDxHat *= invC
        meanDxHatXHat *= invC

        // dx = inv * (dxHat - meanDxHat - xHat * meanDxHatXHat)
        val inv = cache.invStd[i]
        val vInv = FloatVector.broadcast(species, inv)
        val vMeanDx = FloatVector.broadcast(species, meanDxHat)
        val vMeanXX = FloatVector.broadcast(species, meanDxHatXHat)
        j = 0
        while (j < cUpper) {
            val vGy = FloatVector.fromArray(species, gyData, rowOff + j)
            val vG = FloatVector.fromArray(species, gammaData, j)
            val vXh = FloatVector.fromArray(species, xHat, rowOff + j)
            val vDxHat = vGy.mul(vG)
            // term = dxHat - meanDx - xHat*meanXX; dx = inv * term
            val vTerm = vDxHat.sub(vMeanDx).sub(vXh.mul(vMeanXX))
            vTerm.mul(vInv).intoArray(dx, rowOff + j)
            j += laneLen
        }
        while (j < cols) {
            val dxHat = gyData[rowOff + j] * gammaData[j]
            dx[rowOff + j] = inv * (dxHat - meanDxHat - xHat[rowOff + j] * meanDxHatXHat)
            j++
        }
    }
    return dx
}
