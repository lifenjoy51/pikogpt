package train

import gpt.GPTConfig
import kotlinx.serialization.Serializable

// 체크포인트 데이터
@Serializable
data class Checkpoint(
    val modelState: ModelState,
    val optimizerState: OptimizerState,
    val modelArgs: GPTConfig,
    val iterNum: Int,
    val bestValLoss: Double,
    val config: TrainConfig
)