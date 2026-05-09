package turbo

import kotlinx.serialization.Serializable

/**
 * turbo 백엔드 체크포인트 메타데이터. modelArgs가 TurboModelConfig — turbo 전용 옵션도 함께 보존.
 */
@Serializable
data class TurboCheckpoint(
    val iterationNumber: Int,
    val bestValidationLoss: Double,
    val modelArgs: TurboModelConfig,
    val config: TurboTrainConfig,
)
