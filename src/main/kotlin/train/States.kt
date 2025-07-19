package train

import kotlinx.serialization.Serializable

/**
 * Self-Attention 레이어의 상태 정보
 *
 * Transformer의 Multi-Head Self-Attention 메커니즘에서 사용되는
 * Query, Key, Value 프로젝션과 출력 프로젝션 레이어들의 가중치를 저장합니다.
 *
 * @param queryProjection Query 벡터 생성을 위한 선형 레이어 상태
 * @param keyProjection Key 벡터 생성을 위한 선형 레이어 상태
 * @param valueProjection Value 벡터 생성을 위한 선형 레이어 상태
 * @param outputProjection Attention 결과를 출력으로 변환하는 선형 레이어 상태
 */
@Serializable
data class AttentionState(
    val queryProjection: LinearState,
    val keyProjection: LinearState,
    val valueProjection: LinearState,
    val outputProjection: LinearState
)

/**
 * Transformer 블록의 상태 정보
 *
 * 하나의 Transformer 블록에 포함된 모든 레이어들의 상태를 저장합니다.
 * Transformer 아키텍처: LayerNorm -> Self-Attention -> LayerNorm -> FeedForward
 *
 * @param firstLayerNorm 첫 번째 Layer Normalization 레이어 상태
 * @param attention Self-Attention 레이어 상태
 * @param secondLayerNorm 두 번째 Layer Normalization 레이어 상태
 * @param feedForward Feed-Forward Network 레이어 상태
 */
@Serializable
data class BlockState(
    val firstLayerNorm: LayerNormState,
    val attention: AttentionState,
    val secondLayerNorm: LayerNormState,
    val feedForward: FeedForwardState
)

/**
 * Feed-Forward Network의 상태 정보
 *
 * Transformer의 FFN은 두 개의 선형 레이어로 구성됩니다:
 * 1. 확장 레이어: 입력 차원을 4배로 확장
 * 2. 축소 레이어: 다시 원래 차원으로 축소
 *
 * @param fullyConnected 확장 레이어 상태 (embedding_dim -> 4 * embedding_dim)
 * @param projection 축소 레이어 상태 (4 * embedding_dim -> embedding_dim)
 */
@Serializable
data class FeedForwardState(
    val fullyConnected: LinearState,
    val projection: LinearState
)

/**
 * 전체 GPT 모델의 상태 정보
 *
 * GPT 모델의 모든 레이어와 임베딩의 가중치를 저장합니다.
 * 모델 구조: Token Embedding + Position Embedding -> Transformer Blocks -> Language Model Head
 *
 * @param tokenEmbedding 토큰 임베딩 테이블 [vocab_size, embedding_dim]
 * @param positionEmbedding 위치 임베딩 테이블 [block_size, embedding_dim]
 * @param blocks Transformer 블록들의 리스트
 * @param lmHead Language Model Head 레이어 상태 (최종 출력 레이어)
 */
@Serializable
data class ModelState(
    val tokenEmbedding: List<List<Double>>,
    val positionEmbedding: List<List<Double>>,
    val blocks: List<BlockState>,
    val lmHead: LinearState
)

/**
 * Layer Normalization 레이어의 상태 정보
 *
 * Layer Normalization은 입력을 정규화하여 훈련 안정성을 향상시킵니다.
 * 공식: LayerNorm(x) = γ * (x - μ) / σ + β
 *
 * @param weight 스케일 팩터 (γ) - 입력의 각 차원에 대한 스케일링
 * @param bias 편향 팩터 (β) - 입력의 각 차원에 대한 오프셋 (선택적)
 */
@Serializable
data class LayerNormState(
    val weight: List<Double>,
    val bias: List<Double>?
)

/**
 * 선형 레이어의 상태 정보
 *
 * 선형 변환(Linear Transformation)을 수행하는 레이어입니다.
 * 공식: y = xW^T + b
 *
 * @param weight 가중치 행렬 [output_size, input_size]
 * @param bias 편향 벡터 [output_size] (선택적)
 */
@Serializable
data class LinearState(
    val weight: List<List<Double>>,
    val bias: List<Double>?
)

/**
 * 옵티마이저의 상태 정보
 *
 * AdamW 옵티마이저의 내부 상태를 저장합니다.
 * 체크포인트에서 옵티마이저의 상태를 복원하여 정확한 훈련 재개가 가능합니다.
 *
 * @param iteration 현재 이터레이션 번호 (bias correction에 사용)
 * @param firstMoment 1차 모멘트 맵 (그래디언트의 지수 이동 평균)
 * @param secondMoment 2차 모멘트 맵 (그래디언트 제곱의 지수 이동 평균)
 */
@Serializable
data class OptimizerState(
    val iteration: Int,
    val firstMoment: Map<String, List<Double>>, // 1차 모멘트
    val secondMoment: Map<String, List<Double>>  // 2차 모멘트
)
