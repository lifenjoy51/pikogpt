package gpt

import Value
import kotlin.random.Random

/**
 * Dropout 정규화 레이어
 *
 * 과적합을 방지하기 위한 정규화 기법으로, 훈련 중에 랜덤하게 뉴런을 제거합니다.
 * 이를 통해 모델이 특정 뉴런에 과도하게 의존하는 것을 방지하고 일반화 성능을 향상시킵니다.
 *
 * 동작 원리:
 * - 훈련 모드: 지정된 확률로 뉴런을 0으로 설정, 나머지는 스케일링
 * - 추론 모드: 모든 뉴런을 그대로 유지 (드롭아웃 비활성화)
 *
 * @param dropoutProbability 뉴런을 제거할 확률 (0.0~1.0 사이의 값)
 */
class Dropout(
    private val dropoutProbability: Float
) {
    companion object {
        /**
         * 전역 훈련 모드 플래그
         *
         * true: 훈련 모드 - 드롭아웃 활성화
         * false: 추론 모드 - 드롭아웃 비활성화
         *
         * 이 플래그는 훈련과 평가/추론 단계를 구분하기 위해 사용됩니다.
         */
        var training: Boolean = true
    }

    /**
     * 1차원 입력에 대한 Dropout 순전파
     *
     * 각 뉴런에 대해 드롭아웃 확률에 따라 활성화를 결정합니다.
     * 훈련 모드에서만 동작하며, 추론 모드에서는 입력을 그대로 반환합니다.
     *
     * 알고리즘:
     * 1. 훈련 모드가 아니거나 드롭아웃 확률이 0이면 입력 그대로 반환
     * 2. 각 뉴런에 대해 랜덤 숫자 생성
     * 3. 드롭아웃 확률보다 작으면 0으로 설정, 아니면 스케일링 적용
     *
     * @param inputVector 입력 Value 객체들의 1차원 리스트
     * @return Dropout이 적용된 Value 객체들의 리스트
     */
    fun forward(inputVector: List<Value>): List<Value> {
        // 추론 모드이거나 드롭아웃 확률이 0이면 입력 그대로 반환
        if (!training || dropoutProbability <= 0.0f) {
            return inputVector
        }

        // 스케일링 팩터: 드롭아웃된 뉴런들을 보상하기 위해 나머지 뉴런을 증폭
        val compensationScale = 1.0f / (1.0f - dropoutProbability)

        return inputVector.map { neuronValue ->
            if (Random.nextFloat() < dropoutProbability) {
                // 드롭아웃: 뉴런을 0으로 설정
                Value(0.0f)
            } else {
                // 유지: 스케일링 적용하여 기대값 유지
                neuronValue * Value(compensationScale)
            }
        }
    }

    /**
     * 2차원 입력에 대한 Dropout 순전파
     *
     * 시퀀스 데이터에 대해 드롭아웃을 적용합니다.
     * 각 시퀀스 위치의 모든 특징에 대해 독립적으로 드롭아웃을 적용합니다.
     *
     * 사용 예시:
     * - Attention 결과에 대한 정규화
     * - MLP 출력에 대한 정규화
     * - 임베딩 레이어 출력에 대한 정규화
     *
     * @param inputMatrix 2차원 Value 배열 [sequence_length, feature_dimension]
     * @return Dropout이 적용된 2차원 Value 배열
     */
    fun forward(inputMatrix: Array<Array<Value>>): Array<Array<Value>> {
        // 추론 모드이거나 드롭아웃 확률이 0이면 입력 그대로 반환
        if (!training || dropoutProbability <= 0.0f) {
            return inputMatrix
        }

        // 스케일링 팩터: 드롭아웃된 뉴런들을 보상하기 위해 나머지 뉴런을 증폭
        val compensationScale = 1.0f / (1.0f - dropoutProbability)

        return Array(inputMatrix.size) { sequenceIndex ->
            Array(inputMatrix[sequenceIndex].size) { featureIndex ->
                if (Random.nextFloat() < dropoutProbability) {
                    // 드롭아웃: 해당 뉴런을 0으로 설정
                    Value(0.0f)
                } else {
                    // 유지: 스케일링 적용하여 기대값 유지
                    inputMatrix[sequenceIndex][featureIndex] * Value(compensationScale)
                }
            }
        }
    }
}