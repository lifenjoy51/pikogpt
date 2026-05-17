package mps

/**
 * MPSGraph 모델 graph build에 필요한 hyperparam. TurboTrainConfig의 학습 hyperparam과는 분리.
 *
 * graph는 batch/blockSize 등 모든 차원이 빌드 시점에 고정. iter마다 동일 graph 재실행.
 */
data class MpsGraphConfig(
    val numLayers: Int,
    val embedDim: Int,
    val numHeads: Int,
    val blockSize: Int,
    val vocab: Int,
    val batchSize: Int,
    val useSwiglu: Boolean = true,
    val useRope: Boolean = true,
    val tieWeights: Boolean = true,
)
