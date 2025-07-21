package gpt

import Value
import kotlin.math.sqrt

/**
 * 단순화된 Self-Attention 메커니즘
 *
 * Transformer의 핵심 구성 요소로, 입력 시퀀스의 각 위치가 다른 모든 위치와 상호작용하도록 합니다.
 * 언어 모델에서는 인과 마스킹(causal masking)을 사용하여 다음 토큰 예측 시 미래 정보를 보지 못하도록 합니다.
 *
 * 동작 원리:
 * 1. Query(Q), Key(K), Value(V) 행렬 생성
 * 2. Attention Score 계산: QK^T / sqrt(d_k)
 * 3. Causal Mask 적용 (미래 토큰 마스킹)
 * 4. Softmax로 어텐션 가중치 계산
 * 5. Value와 가중 합 계산
 * 6. 출력 프로젝션 및 Dropout 적용
 *
 * @param modelConfig GPT 모델 설정 (어텐션 헤드 수, 임베딩 차원 등)
 */
class SimpleSelfAttention(private val modelConfig: GPTConfig) {
    /** 각 어텐션 헤드의 차원 (embedding_dim / num_heads) */
    private val attentionHeadDimension = modelConfig.nEmbd / modelConfig.nHead

    /** Query 행렬 생성을 위한 선형 프로젝션 레이어 */
    private val queryProjection = Linear(modelConfig.nEmbd, modelConfig.nEmbd, modelConfig.bias)

    /** Key 행렬 생성을 위한 선형 프로젝션 레이어 */
    private val keyProjection = Linear(modelConfig.nEmbd, modelConfig.nEmbd, modelConfig.bias)

    /** Value 행렬 생성을 위한 선형 프로젝션 레이어 */
    private val valueProjection = Linear(modelConfig.nEmbd, modelConfig.nEmbd, modelConfig.bias)

    /** 출력 프로젝션 레이어 (어텐션 결과를 최종 출력으로 변환) */
    private val outputProjection = Linear(modelConfig.nEmbd, modelConfig.nEmbd, modelConfig.bias)

    /** 정규화를 위한 Dropout 레이어 */
    private val attentionDropout = Dropout(modelConfig.dropout)

    /**
     * Self-Attention 메커니즘 순전파
     *
     * 입력 시퀀스에 Self-Attention을 적용하여 각 위치에서 다른 모든 위치의 정보를 고려합니다.
     * Causal masking을 사용하여 언어 모델의 자기회귀적 특성을 유지합니다.
     *
     * 계산 단계:
     * 1. 입력에서 Q, K, V 행렬 생성
     * 2. Scaled Dot-Product Attention 점수 계산
     * 3. Causal mask 적용 (미래 정보 차단)
     * 4. Softmax로 어텐션 가중치 정규화
     * 5. Value 행렬과 가중 합 계산
     * 6. 출력 프로젝션 및 Dropout 적용
     *
     * @param inputSequence 입력 시퀀스
     * @return 어텐션이 적용된 출력 시퀀스
     */
    fun forward(inputSequence: Sequence): Sequence {
        val tokenCount = inputSequence.tokenCount
        // 단순화를 위해 배치 크기를 1로 고정 (실제 구현에서는 배치 처리 필요)

        // 1. Query, Key, Value 행렬 생성
        val queryMatrix = inputSequence.mapTokens { sequenceElement ->
            queryProjection.forward(sequenceElement)
        }

        val keyMatrix = inputSequence.mapTokens { sequenceElement ->
            keyProjection.forward(sequenceElement)
        }

        val valueMatrix = inputSequence.mapTokens { sequenceElement ->
            valueProjection.forward(sequenceElement)
        }

        // 2. Scaled Dot-Product Attention 점수 계산
        // 스케일 팩터: 1/sqrt(head_dimension) - 어텐션 점수의 분산을 안정화
        val attentionScale = Value(1.0f / sqrt(attentionHeadDimension.toFloat()))

        val attentionScores = Matrix.fromArray(Array(tokenCount) { queryIndex ->
            Array(tokenCount) { keyIndex ->
                if (keyIndex <= queryIndex) { // Causal mask: 현재 이전 위치만 참조 가능
                    // 닷젝곱 계산: Q[i] · K[j]
                    var dotProduct = Value(0.0f)
                    for (embeddingIndex in 0 until modelConfig.nEmbd) {
                        dotProduct = dotProduct + queryMatrix[queryIndex][embeddingIndex] * keyMatrix[keyIndex][embeddingIndex]
                    }
                    dotProduct * attentionScale
                } else {
                    // 미래 위치는 매우 작은 값으로 마스킹 (소프트맥스 후 거의 0이 됨)
                    Value(-1e9f)
                }
            }
        })

        // 3. Softmax 정규화 (각 행별로 수행)
        val normalizedAttentionWeights = attentionScores.mapRows { scoreRow ->
            // 수치 안정성을 위해 최대값을 뺄
            val maxScore = scoreRow.maxByOrNull { it.scalarValue } ?: Value(0.0f)
            val exponentialScores = scoreRow.map { score -> (score - maxScore).exp() }.toTypedArray()
            val sumOfExponentials = exponentialScores.reduce { accumulator, expScore -> accumulator + expScore }
            exponentialScores.map { expScore -> expScore / sumOfExponentials }.toTypedArray()
        }

        // 4. Value 행렬과 어텐션 가중치의 가중 합 계산
        val attentionOutputArray = Array(tokenCount) { queryIndex ->
            Array(modelConfig.nEmbd) { embeddingIndex ->
                var weightedSum = Value(0.0f)
                for (keyIndex in 0 until tokenCount) {
                    weightedSum = weightedSum + normalizedAttentionWeights[queryIndex][keyIndex] * valueMatrix[keyIndex][embeddingIndex]
                }
                weightedSum
            }
        }

        // 5. 출력 프로젝션 및 Dropout 적용
        val outputSequence = Sequence.fromArray(attentionOutputArray).mapTokens { attentionVector ->
            outputProjection.forward(attentionVector)
        }

        return attentionDropout.forward(outputSequence)
    }

    /**
     * Self-Attention 레이어의 모든 학습 가능한 파라미터 수집
     *
     * Q, K, V 프로젝션 레이어와 출력 프로젝션 레이어의 모든 파라미터를 포함합니다.
     * 이 파라미터들은 어텐션 메커니즘이 시퀀스 간 상호작용을 학습하는 데 필수적입니다.
     *
     * @return 모든 학습 가능한 Value 객체들의 리스트
     */
    fun parameters(): List<Value> {
        return queryProjection.parameters() + keyProjection.parameters() +
            valueProjection.parameters() + outputProjection.parameters()
    }
}