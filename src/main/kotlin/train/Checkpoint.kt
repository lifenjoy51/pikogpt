package train

import gpt.GPTConfig
import kotlinx.serialization.Serializable

/**
 * 체크포인트 데이터 클래스
 *
 * 훈련 중인 모델의 상태를 저장하고 로드하기 위한 데이터 구조체입니다.
 * 이를 통해 훈련을 중단하고 나중에 재개하거나, 최고 성능의 모델을 보존할 수 있습니다.
 *
 * @param modelState 모델의 가중치와 편향을 포함하는 상태 정보
 * @param optimizerState 옵티마이저의 내부 상태 (모멘트, 이터레이션 등)
 * @param modelArgs 모델 아키텍처 설정 (GPT 설정)
 * @param iterationNumber 현재 훈련 이터레이션 번호
 * @param bestValidationLoss 지금까지의 최고 검증 손실 값
 * @param config 훈련 설정 매개변수들
 */
@Serializable
data class Checkpoint(
    val modelState: ModelState,
    val optimizerState: OptimizerState,
    val modelArgs: GPTConfig,
    val iterationNumber: Int,
    val bestValidationLoss: Double,
    val config: TrainConfig
)