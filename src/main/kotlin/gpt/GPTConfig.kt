package gpt

import kotlinx.serialization.Serializable

/**
 * GPT 모델 아키텍처 설정
 *
 * GPT 모델의 구조를 정의하는 모든 하이퍼파라미터를 포함합니다.
 * 이 설정들은 모델의 크그, 복잡도, 성능을 결정합니다.
 *
 * @param blockSize 시퀀스 최대 길이 (맥락 윈도우) - 모델이 한 번에 처리할 수 있는 토큰 수
 * @param vocabSize 어휘 사전 크기 - 모델이 알고 있는 가능한 모든 토큰의 수
 * @param nLayer Transformer 블록 수 - 모델의 깊이를 결정하는 중요한 요소
 * @param nHead Multi-Head Attention의 헤드 수 - 병렬 어텐션 메커니즘 수
 * @param nEmbd 임베딩 차원 - 모델의 표현력과 직결되는 핵심 파라미터
 * @param bias 선형 레이어에 편향(bias) 사용 여부
 * @param dropout Dropout 확률 - 과적합 방지를 위한 정규화 기법
 */
@Serializable
data class GPTConfig(
    /** 최대 시퀀스 길이 - 모델이 한 번에 처리할 수 있는 토큰 수 */
    val maxSequenceLength: Int,

    /** 어휘 사전 크기 - 모델이 예측할 수 있는 총 토큰 수 */
    val vocabularySize: Int,

    /** Transformer 레이어 수 - 모델의 깊이를 결정 */
    val numberOfLayers: Int,

    /** Multi-Head Attention 헤드 수 - 병렬 어텐션 메커니즘 수 */
    val numberOfAttentionHeads: Int,

    /** 임베딩 차원 - 모든 벡터 표현의 기본 차원 */
    val embeddingDimension: Int,

    /** 편향 사용 여부 - 선형 레이어에서 bias term 포함 여부 */
    val useBias: Boolean,

    /** 드롭아웃 확률 - 정규화를 위한 뉴런 제거 비율 */
    val dropoutProbability: Float
) {
    // 호환성을 위한 별칭 속성들
    val blockSize: Int get() = maxSequenceLength
    val vocabSize: Int get() = vocabularySize
    val nLayer: Int get() = numberOfLayers
    val nHead: Int get() = numberOfAttentionHeads
    val nEmbd: Int get() = embeddingDimension
    val bias: Boolean get() = useBias
    val dropout: Float get() = dropoutProbability
}