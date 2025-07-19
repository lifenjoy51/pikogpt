package gpt

import Value

/**
 * Multi-Layer Perceptron (MLP) - Feed-Forward Network
 *
 * Transformer 블록의 두 번째 주요 구성 요소로, 비선형 변환을 수행합니다.
 * 어텐션 메커니즘에서 나온 표현을 더 다양하고 복잡한 표현으로 변환합니다.
 *
 * MLP 구조:
 * 1. 확장 선형 레이어: embedding_dim → 4 * embedding_dim
 * 2. GELU 활성화 함수: 비선형성 도입
 * 3. 수축 선형 레이어: 4 * embedding_dim → embedding_dim
 * 4. Dropout: 정규화를 위한 래덤 뉴런 제거
 *
 * 이 구조는 GPT 논문에서 제안된 표준 비율(4x expansion)을 따릅니다.
 *
 * @param modelConfig GPT 모델 설정 (임베딩 차원, 드롭아웃 비율 등)
 */
class MLP(modelConfig: GPTConfig) {
    /** 확장 레이어: 임베딩 차원을 4배로 확장 */
    private val expansionLayer = Linear(modelConfig.nEmbd, 4 * modelConfig.nEmbd, modelConfig.bias)

    /** 수축 레이어: 4배 확장된 차원을 원래 차원으로 복원 */
    private val contractionLayer = Linear(4 * modelConfig.nEmbd, modelConfig.nEmbd, modelConfig.bias)

    /** 정규화를 위한 Dropout 레이어 */
    private val mlpDropout = Dropout(modelConfig.dropout)

    /**
     * MLP Feed-Forward Network 순전파
     *
     * 입력 벡터를 비선형 변환을 통해 더 표현력 있는 특징으로 변환합니다.
     * 이는 Transformer의 핑트 오는 단계로, 어텐션에서 나온 문맥 정보를 더 세밀하게 가공합니다.
     *
     * 처리 단계:
     * 1. 확장: 임베딩 차원을 4배로 늘려 더 많은 표현력 확보
     * 2. 활성화: GELU 함수로 비선형성 도입
     * 3. 수축: 원래 임베딩 차원으로 복원
     * 4. 정규화: Dropout으로 과적합 방지
     *
     * @param inputVector 입력 벡터 [embedding_dimension]
     * @return 변환된 출력 벡터 [embedding_dimension]
     */
    fun forward(inputVector: Array<Value>): Array<Value> {
        // 1. 확장 선형 변환 (embedding_dim → 4 * embedding_dim)
        var transformedVector = expansionLayer.forward(inputVector)

        // 2. GELU 활성화 함수 적용 (비선형성 도입)
        transformedVector = transformedVector.map { neuronOutput ->
            neuronOutput.gelu()
        }.toTypedArray()

        // 3. 수축 선형 변환 (4 * embedding_dim → embedding_dim)
        transformedVector = contractionLayer.forward(transformedVector)

        // 4. Dropout 적용 (정규화)
        return mlpDropout.forward(transformedVector.toList()).toTypedArray()
    }

    /**
     * MLP의 모든 학습 가능한 파라미터 수집
     *
     * 확장 레이어와 수축 레이어의 모든 가중치와 편향을 포함합니다.
     * 이 파라미터들은 MLP가 입력 표현을 더 복잡하고 유용한 표현으로 변환하는 데 필수적입니다.
     *
     * @return 모든 학습 가능한 Value 객체들의 리스트
     */
    fun parameters(): List<Value> {
        return expansionLayer.parameters() + contractionLayer.parameters()
    }
}