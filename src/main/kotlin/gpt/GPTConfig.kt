package gpt

import kotlinx.serialization.Serializable

/**
 * GPT 모델 아키텍처 설정 (scalar/turbo 백엔드 공유).
 *
 * 두 백엔드가 모두 사용하는 기본 모델 형태(layers/heads/embd/dropout/bias) 만 담는다.
 * Turbo 전용 옵션(weight tying, MLP activation 종류, position encoding 종류 등)은
 * [turbo.TurboModelConfig]에 분리되어 있다.
 *
 * @param maxSequenceLength 시퀀스 최대 길이 (맥락 윈도우)
 * @param vocabularySize 어휘 사전 크기
 * @param numberOfLayers Transformer 블록 수
 * @param numberOfAttentionHeads Multi-Head Attention 헤드 수
 * @param embeddingDimension 임베딩 차원
 * @param useBias 선형 레이어 bias 사용 여부
 * @param dropoutProbability Dropout 확률
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
    val dropoutProbability: Float,
)
