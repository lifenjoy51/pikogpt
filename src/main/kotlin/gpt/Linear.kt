package gpt

import RandomGaussian
import Value

/**
 * 선형 변환 레이어 (완전 연결 레이어)
 *
 * 신경망의 기본 구성 요소로, 입력 벡터를 선형 변환하여 출력 벡터를 생성합니다.
 * y = xW^T + b (W: 가중치 행렬, b: 편향 벡터)
 *
 * 이 레이어는 다음과 같은 용도로 사용됩니다:
 * - MLP(Multi-Layer Perceptron)의 구성 요소
 * - Attention 메커니즘의 Query/Key/Value 투영
 * - 최종 언어 모델 헤드 (어휘 확률 분포 생성)
 *
 * @param inputFeatureCount 입력 벡터의 차원 수
 * @param outputFeatureCount 출력 벡터의 차원 수
 * @param enableBias 편향(bias) 사용 여부 (기본값: true)
 */
class Linear(
    inputFeatureCount: Int,
    private val outputFeatureCount: Int,
    enableBias: Boolean = true
) {
    /**
     * 가중치 행렬 [outputFeatureCount, inputFeatureCount]
     *
     * 각 가중치는 표준편차 0.02의 가우시안 분포로 초기화됩니다.
     * 이는 Xavier/He 초기화 방법의 변형으로, 작은 값으로 시작하여 훈련 안정성을 돕습니다.
     */
    private val weightMatrix = Array(outputFeatureCount) {
        Array(inputFeatureCount) { Value((RandomGaussian.next() * 0.02).toFloat()) }
    }

    /**
     * 편향 벡터 [outputFeatureCount]
     *
     * 선형 변환에 추가되는 학습 가능한 편향 항입니다.
     * 0으로 초기화되며, enableBias가 false인 경우 null입니다.
     */
    private val biasVector = if (enableBias) Array(outputFeatureCount) { Value(0.0f) } else null

    /**
     * 선형 변환 순전파
     *
     * 입력 벡터에 가중치 행렬을 곱하고 편향을 더하여 출력을 계산합니다.
     * 수식: output[i] = sum(input[j] * weight[i][j]) + bias[i]
     *
     * 계산 과정:
     * 1. 각 출력 뉴런에 대해 가중합(weighted sum) 계산
     * 2. 편향이 활성화된 경우 편향 값 추가
     * 3. 최종 출력 벡터 반환
     *
     * @param inputVector 입력 벡터 [inputFeatureCount]
     * @return 변환된 출력 벡터 [outputFeatureCount]
     */
    fun forward(inputVector: Array<Value>): Array<Value> {
        return Array(outputFeatureCount) { outputIndex ->
            // 각 출력 뉴런에 대한 가중합 계산
            val weightedSum = inputVector.zip(weightMatrix[outputIndex]) { inputValue, weightValue ->
                inputValue * weightValue
            }.reduce { accumulator, product -> accumulator + product }

            // 편향이 있는 경우 추가, 없으면 가중합만 반환
            biasVector?.get(outputIndex)?.let { biasValue ->
                weightedSum + biasValue
            } ?: weightedSum
        }
    }

    /**
     * 레이어의 모든 학습 가능한 파라미터 수집
     *
     * 옵티마이저가 그래디언트를 적용할 수 있도록 모든 파라미터를 하나의 리스트로 반환합니다.
     * 가중치 행렬의 모든 원소와 편향 벡터(활성화된 경우)를 포함합니다.
     *
     * @return 모든 학습 가능한 Value 객체들의 리스트
     */
    fun parameters(): List<Value> {
        // 가중치 행렬을 1차원으로 평면화하고, 편향 벡터가 있으면 추가
        return weightMatrix.flatten() + (biasVector?.toList() ?: emptyList())
    }
}