package vec

import gpt.GPTConfig
import kotlinx.serialization.Serializable
import train.TrainConfig

/**
 * 벡터 백엔드 체크포인트의 메타데이터.
 *
 * 스칼라 `train.ScalarCheckpoint`와 달리 `ModelState`(더미 가중치 JSON)는 두지 않는다.
 * 실제 가중치는 [VecTrainer]가 `model_weights.bin`에 연속 float32 big-endian로 저장하며,
 * 이 JSON은 아키텍처·학습 상태 메타만 담는다.
 */
@Serializable
data class VecCheckpoint(
    val iterationNumber: Int,
    val bestValidationLoss: Double,
    val modelArgs: GPTConfig,
    val config: TrainConfig,
)
