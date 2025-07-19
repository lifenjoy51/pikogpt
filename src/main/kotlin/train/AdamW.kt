package train

import Value
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * AdamW 옵티마이저 구현
 *
 * AdamW는 Adam 옵티마이저에 가중치 감쇠(weight decay)를 개선한 방식으로 적용한 옵티마이저입니다.
 * 기존 Adam의 L2 정규화 방식과 달리, 가중치 감쇠를 그래디언트 업데이트와 분리하여 적용합니다.
 *
 * @param parameters 최적화할 파라미터들의 리스트 (Value 객체들)
 * @param learningRate 학습률 (기본값: 0.001)
 * @param beta1 1차 모멘트 지수 감쇠율 (기본값: 0.9)
 * @param beta2 2차 모멘트 지수 감쇠율 (기본값: 0.999)
 * @param weightDecay 가중치 감쇠 계수 (기본값: 0.01)
 * @param epsilon 수치 안정성을 위한 작은 값 (기본값: 1e-8)
 */
class AdamW(
    private val parameters: List<Value>,
    private var learningRate: Float = 1e-3f,
    private val beta1: Float = 0.9f,
    private val beta2: Float = 0.999f,
    private val weightDecay: Float = 0.01f,
    private val epsilon: Float = 1e-8f
) {
    /** 1차 모멘트 (그래디언트의 지수 이동 평균) - 그래디언트의 방향성을 기억 */
    private val firstMoment = mutableMapOf<Value, Float>()

    /** 2차 모멘트 (그래디언트 제곱의 지수 이동 평균) - 그래디언트의 크기를 기억 */
    private val secondMoment = mutableMapOf<Value, Float>()

    /** 현재 타임스텝 (bias correction에 사용) */
    private var timeStep = 0

    /**
     * 옵티마이저 초기화
     * 모든 파라미터에 대해 모멘트 값들을 0으로 초기화합니다.
     */
    init {
        parameters.forEach { parameter ->
            firstMoment[parameter] = 0.0f
            secondMoment[parameter] = 0.0f
        }
    }

    /**
     * 옵티마이저 스텝 실행
     *
     * AdamW 알고리즘에 따라 파라미터를 업데이트합니다:
     * 1. 가중치 감쇠 적용 (AdamW의 핵심 개선사항)
     * 2. 1차, 2차 모멘트 업데이트
     * 3. Bias correction 적용
     * 4. 최종 파라미터 업데이트
     */
    fun step() {
        timeStep++

        parameters.forEach { parameter ->
            // 그래디언트가 0인 파라미터는 건너뜀
            if (parameter.gradient == 0.0f) return@forEach

            // AdamW 가중치 감쇠: 그래디언트와 독립적으로 적용
            // 이는 Adam의 L2 정규화와 다른 점으로, 더 나은 일반화 성능을 제공
            if (weightDecay > 0) {
                parameter.scalarValue -= learningRate * weightDecay * parameter.scalarValue
            }

            // Adam 모멘트 업데이트
            val gradient = parameter.gradient

            // 1차 모멘트: 그래디언트의 지수 이동 평균 (방향성)
            firstMoment[parameter] = beta1 * firstMoment[parameter]!! + (1.0f - beta1) * gradient

            // 2차 모멘트: 그래디언트 제곱의 지수 이동 평균 (크기)
            secondMoment[parameter] = beta2 * secondMoment[parameter]!! + (1.0f - beta2) * gradient * gradient

            // Bias correction: 초기 스텝에서의 편향 보정
            // 모멘트들이 0으로 초기화되어 발생하는 편향을 보정
            val biasCorrectFirstMoment = firstMoment[parameter]!! / (1 - beta1.pow(timeStep))
            val biasCorrectSecondMoment = secondMoment[parameter]!! / (1 - beta2.pow(timeStep))

            // 최종 파라미터 업데이트
            // 1차 모멘트(방향)를 2차 모멘트의 제곱근(적응적 학습률)로 나누어 업데이트
            parameter.scalarValue -= learningRate * biasCorrectFirstMoment / (sqrt(biasCorrectSecondMoment) + epsilon)
        }
    }

    /**
     * 모든 파라미터의 그래디언트를 0으로 초기화
     *
     * PyTorch와 같은 자동 미분 프레임워크에서는 그래디언트가 누적되므로,
     * 매 배치마다 그래디언트를 초기화해야 합니다.
     */
    fun zeroGrad() {
        parameters.forEach { it.gradient = 0.0f }
    }

    /**
     * 학습률 업데이트
     *
     * 학습률 스케줄링(learning rate scheduling)을 위해 사용됩니다.
     * 일반적으로 훈련 중에 학습률을 점진적으로 감소시키거나 특정 패턴으로 조정합니다.
     *
     * @param newLearningRate 새로운 학습률 값
     */
    fun updateLearningRate(newLearningRate: Float) {
        learningRate = newLearningRate
    }
}