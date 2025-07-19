package gpt

import Value

/**
 * Layer Normalization 레이어
 *
 * 층 정규화는 각 최인에 대해 활성화 값을 정규화하여 훈련 안정성을 향상시킵니다.
 * 비선형 활성화 함수 전에 적용하여 그래디언트 폭발/소실 문제를 완화합니다.
 *
 * 수식: LayerNorm(x) = γ * (x - μ) / σ + β
 * - μ: 평균 (mean)
 * - σ: 표준편차 (standard deviation)
 * - γ: 학습 가능한 스케일 파라미터 (weight)
 * - β: 학습 가능한 시프트 파라미터 (bias)
 *
 * @param featureDimension 정규화할 특징 차원 수
 * @param enableBias 편향 파라미터 사용 여부
 */
class LayerNorm(
    featureDimension: Int,
    enableBias: Boolean
) {
    /**
     * 스케일 파라미터 (감마, γ) [featureDimension]
     * 정규화된 값에 곱해지는 학습 가능한 가중치로, 1.0으로 초기화됩니다.
     */
    private val scaleParameters = Array(featureDimension) { Value(1.0f) }

    /**
     * 시프트 파라미터 (베타, β) [featureDimension]
     * 정규화된 값에 더해지는 학습 가능한 편향으로, 0.0으로 초기화됩니다.
     */
    private val shiftParameters = if (enableBias) Array(featureDimension) { Value(0.0f) } else null

    /**
     * Layer Normalization 순전파
     *
     * 입력 벡터에 종 정규화를 적용합니다.
     * 전체 입력에 대해 평균과 분산을 계산하고, 이를 사용하여 정규화합니다.
     *
     * 계산 단계:
     * 1. 평균 계산: μ = (1/n) * ∑x_i
     * 2. 분산 계산: σ² = (1/n) * ∑(x_i - μ)²
     * 3. 표준화: (x_i - μ) / sqrt(σ² + ε)
     * 4. 스케일링 및 시프트: γ * normalized + β
     *
     * @param inputVector 정규화할 입력 벡터 [featureDimension]
     * @return 정규화된 출력 벡터 [featureDimension]
     */
    fun forward(inputVector: Array<Value>): Array<Value> {
        val featureCount = inputVector.size.toFloat()

        // 1. 평균 계산
        val featureMean = inputVector.reduce { accumulator, value -> accumulator + value } / featureCount

        // 2. 분산 계산
        val featureVariance = inputVector.map { featureValue ->
            (featureValue - featureMean).pow(2.0f)
        }.reduce { accumulator, squaredDiff -> accumulator + squaredDiff } / featureCount

        // 3. 수치 안정성을 위한 엡실론 및 표준편차 역수
        val epsilon = Value(1e-5f)  // 나눗셈에서 0을 방지하기 위한 작은 상수
        val inverseStandardDeviation = (featureVariance + epsilon).pow(-0.5f)

        // 4. 정규화, 스케일링, 시프트 적용
        return inputVector.mapIndexed { featureIndex, originalValue ->
            // 정규화: (x - 마이너스 평균) / 표준편차
            val normalizedValue = (originalValue - featureMean) * inverseStandardDeviation

            // 스케일 적용
            val scaledValue = normalizedValue * scaleParameters[featureIndex]

            // 시프트 적용 (편향이 있는 경우)
            shiftParameters?.get(featureIndex)?.let { biasValue ->
                scaledValue + biasValue
            } ?: scaledValue
        }.toTypedArray()
    }

    /**
     * Layer Normalization의 모든 학습 가능한 파라미터 수집
     *
     * 스케일 파라미터(γ)와 시프트 파라미터(β)를 포함합니다.
     * 옵티마이저가 이 파라미터들에 그래디언트를 적용하여 모델을 학습시킵니다.
     *
     * @return 모든 학습 가능한 Value 객체들의 리스트
     */
    fun parameters(): List<Value> {
        return scaleParameters.toList() + (shiftParameters?.toList() ?: emptyList())
    }
}