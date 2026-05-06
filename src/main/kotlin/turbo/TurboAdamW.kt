package turbo

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import java.io.File
import java.nio.ByteBuffer
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * AdamW 옵티마이저. Phase 2부터 SIMD 단일 패스 fused 루프 — m/v/decay/p를
 * 한 lane register pass에서 모두 갱신해 메모리 통과 횟수 최소화.
 *
 *   1) decoupled weight decay: p ← p · (1 - lr·wd)
 *   2) m ← β1·m + (1-β1)·g
 *   3) v ← β2·v + (1-β2)·g²
 *   4) m̂ = m / (1 - β1^t),  v̂ = v / (1 - β2^t)
 *   5) p ← p - lr · m̂ / (√v̂ + ε)
 */
class TurboAdamW(
    private val parameters: List<TurboTensor>,
    private var learningRate: Float = 1e-3f,
    private val beta1: Float = 0.9f,
    private val beta2: Float = 0.999f,
    private val weightDecay: Float = 0.01f,
    private val epsilon: Float = 1e-8f,
) {
    private val firstMoment: List<FloatArray> = parameters.map { FloatArray(it.numel) }
    private val secondMoment: List<FloatArray> = parameters.map { FloatArray(it.numel) }
    private var timeStep: Int = 0

    fun step() {
        timeStep++
        val bc1 = 1.0f - beta1.pow(timeStep)
        val bc2 = 1.0f - beta2.pow(timeStep)
        val invBc1 = 1.0f / bc1
        val invBc2 = 1.0f / bc2
        val decayCoeff = 1.0f - learningRate * weightDecay
        val negLr = -learningRate

        val species = TurboSimdMath.SPECIES
        val laneLen = species.length()
        val vBeta1 = FloatVector.broadcast(species, beta1)
        val vOneMinusB1 = FloatVector.broadcast(species, 1.0f - beta1)
        val vBeta2 = FloatVector.broadcast(species, beta2)
        val vOneMinusB2 = FloatVector.broadcast(species, 1.0f - beta2)
        val vInvBc1 = FloatVector.broadcast(species, invBc1)
        val vInvBc2 = FloatVector.broadcast(species, invBc2)
        val vEps = FloatVector.broadcast(species, epsilon)
        val vDecay = FloatVector.broadcast(species, decayCoeff)
        val vNegLr = FloatVector.broadcast(species, negLr)

        for (pIdx in parameters.indices) {
            val p = parameters[pIdx]
            val grad = p.grad ?: continue
            val m = firstMoment[pIdx]
            val v = secondMoment[pIdx]
            val data = p.data
            val len = data.size
            val upper = species.loopBound(len)

            var i = 0
            while (i < upper) {
                val vG = FloatVector.fromArray(species, grad, i)
                val vMOld = FloatVector.fromArray(species, m, i)
                val vVOld = FloatVector.fromArray(species, v, i)

                // m ← β1·m + (1-β1)·g
                val vM = vMOld.mul(vBeta1).add(vG.mul(vOneMinusB1))
                // v ← β2·v + (1-β2)·g²
                val vV = vVOld.mul(vBeta2).add(vG.mul(vG).mul(vOneMinusB2))
                vM.intoArray(m, i)
                vV.intoArray(v, i)

                // update = m̂ / (√v̂ + ε)
                val vMHat = vM.mul(vInvBc1)
                val vVHat = vV.mul(vInvBc2)
                val vDenom = vVHat.lanewise(VectorOperators.SQRT).add(vEps)
                val vUpdate = vMHat.div(vDenom)

                // p ← p · decayCoeff + (-lr) · update
                val vP = FloatVector.fromArray(species, data, i)
                vUpdate.fma(vNegLr, vP.mul(vDecay)).intoArray(data, i)

                i += laneLen
            }
            while (i < len) {
                val g = grad[i]
                val mNew = beta1 * m[i] + (1.0f - beta1) * g
                val vNew = beta2 * v[i] + (1.0f - beta2) * g * g
                m[i] = mNew
                v[i] = vNew
                val mHat = mNew * invBc1
                val vHat = vNew * invBc2
                data[i] = data[i] * decayCoeff - learningRate * mHat / (sqrt(vHat) + epsilon)
                i++
            }
        }
    }

    fun zeroGrad() {
        for (p in parameters) p.zeroGrad()
    }

    fun updateLearningRate(newLearningRate: Float) {
        learningRate = newLearningRate
    }

    fun resetState() {
        timeStep = 0
        for (m in firstMoment) m.fill(0.0f)
        for (v in secondMoment) v.fill(0.0f)
    }

    fun saveState(file: File) {
        file.outputStream().use { out ->
            out.write(ByteBuffer.allocate(4).putInt(timeStep).array())
            for (pIdx in parameters.indices) {
                val m = firstMoment[pIdx]
                val v = secondMoment[pIdx]
                val buf = ByteBuffer.allocate((m.size + v.size) * 4)
                for (x in m) buf.putFloat(x)
                for (x in v) buf.putFloat(x)
                out.write(buf.array())
            }
        }
    }

    fun loadState(file: File) {
        file.inputStream().use { input ->
            val hdr = ByteArray(4)
            require(input.read(hdr) == 4) { "옵티마이저 상태 파일 EOF 조기 도달 (header)" }
            timeStep = ByteBuffer.wrap(hdr).int

            val buf = ByteArray(4)
            for (pIdx in parameters.indices) {
                val m = firstMoment[pIdx]
                for (i in m.indices) {
                    require(input.read(buf) == 4) { "옵티마이저 상태 파일 EOF 조기 도달 (m, param=$pIdx, i=$i)" }
                    m[i] = ByteBuffer.wrap(buf).float
                }
                val v = secondMoment[pIdx]
                for (i in v.indices) {
                    require(input.read(buf) == 4) { "옵티마이저 상태 파일 EOF 조기 도달 (v, param=$pIdx, i=$i)" }
                    v[i] = ByteBuffer.wrap(buf).float
                }
            }
        }
    }
}
