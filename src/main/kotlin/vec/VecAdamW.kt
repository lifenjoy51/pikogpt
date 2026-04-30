package vec

import java.io.File
import java.nio.ByteBuffer
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * AdamW 옵티마이저의 벡터(Tensor) 버전.
 *
 * 스칼라 `train.AdamW`와 수식은 동일하지만, 파라미터가 `Value` 하나씩이 아니라
 * `Tensor` 단위라 1차/2차 모멘트를 Tensor별 FloatArray로 관리한다.
 *
 *   1) 가중치 감쇠 (decoupled): p -= lr * wd * p
 *   2) m = β1·m + (1-β1)·g
 *      v = β2·v + (1-β2)·g²
 *   3) bias correction: m̂ = m / (1 - β1^t),  v̂ = v / (1 - β2^t)
 *   4) p -= lr · m̂ / (√v̂ + ε)
 *
 * 학습 루프가 [step] 호출 전에 grad를 채워 두고, 이후 [zeroGrad]로 초기화하는
 * 관례를 따른다.
 */
class VecAdamW(
    private val parameters: List<Tensor>,
    private var learningRate: Float = 1e-3f,
    private val beta1: Float = 0.9f,
    private val beta2: Float = 0.999f,
    private val weightDecay: Float = 0.01f,
    private val epsilon: Float = 1e-8f,
) {
    /** 파라미터 인덱스별 1차 모멘트 (FloatArray, 해당 param과 같은 크기). */
    private val firstMoment: List<FloatArray> = parameters.map { FloatArray(it.numel) }

    /** 파라미터 인덱스별 2차 모멘트. */
    private val secondMoment: List<FloatArray> = parameters.map { FloatArray(it.numel) }

    /** 현재 optimizer step 수 — bias correction에 사용. */
    private var timeStep: Int = 0

    fun step() {
        timeStep++
        val bc1 = 1.0f - beta1.pow(timeStep)
        val bc2 = 1.0f - beta2.pow(timeStep)

        for (pIdx in parameters.indices) {
            val p = parameters[pIdx]
            val grad = p.grad ?: continue  // grad 없으면 업데이트 안 함
            val m = firstMoment[pIdx]
            val v = secondMoment[pIdx]
            val data = p.data

            for (i in data.indices) {
                val g = grad[i]

                // 1) weight decay — decoupled (AdamW 핵심)
                if (weightDecay > 0) {
                    data[i] -= learningRate * weightDecay * data[i]
                }

                // 2) 모멘트 업데이트
                m[i] = beta1 * m[i] + (1.0f - beta1) * g
                v[i] = beta2 * v[i] + (1.0f - beta2) * g * g

                // 3) bias correction
                val mHat = m[i] / bc1
                val vHat = v[i] / bc2

                // 4) 파라미터 업데이트
                data[i] -= learningRate * mHat / (sqrt(vHat) + epsilon)
            }
        }
    }

    /** 모든 파라미터의 grad FloatArray를 0으로. */
    fun zeroGrad() {
        for (p in parameters) p.zeroGrad()
    }

    /** LR 스케줄러에서 호출. */
    fun updateLearningRate(newLearningRate: Float) {
        learningRate = newLearningRate
    }

    /**
     * Optimizer 내부 상태(timeStep + 1·2차 모멘트)를 0으로 초기화.
     *
     * Pretrain 가중치를 로드한 뒤 finetune을 시작할 때 호출 — 이전 학습의 모멘텀이
     * 새 데이터 분포에 부적절할 수 있으므로 깨끗하게 reset.
     */
    fun resetState() {
        timeStep = 0
        for (m in firstMoment) m.fill(0.0f)
        for (v in secondMoment) v.fill(0.0f)
    }

    /**
     * 옵티마이저 상태(timeStep + 모든 파라미터의 1·2차 모멘트)를 binary로 저장.
     *
     * 포맷: `[int32 timeStep][param0 m][param0 v][param1 m][param1 v]...`
     * 모든 float/int는 big-endian. `[paramK m/v]`는 해당 파라미터 크기(numel)의 float32 연속.
     *
     * 파라미터 개수/크기는 모델 아키텍처에 의해 결정되므로, 로드 측이 같은 모델로
     * 옵티마이저를 만든 뒤 이 파일을 읽으면 그대로 복원된다.
     */
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

    /** [saveState]로 저장된 파일에서 옵티마이저 상태를 복원. 파라미터 shape가 일치해야 한다. */
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
