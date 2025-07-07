package train

import kotlinx.serialization.Serializable

// 훈련 설정
@Serializable
data class TrainConfig(
    // I/O
    val outDir: String = "out-shakespeare-char",
    val evalInterval: Int = 250,
    val logInterval: Int = 10,
    val evalIters: Int = 200,
    val evalOnly: Boolean = false,
    val alwaysSaveCheckpoint: Boolean = false,
    val initFrom: String = "scratch", // 'scratch' or 'resume'

    // 데이터
    val dataset: String = "shakespeare_char",
    val gradientAccumulationSteps: Int = 1,
    val batchSize: Int = 64,
    val blockSize: Int = 256,

    // 모델
    val nLayer: Int = 6,
    val nHead: Int = 6,
    val nEmbd: Int = 384,
    val dropout: Double = 0.2,
    val bias: Boolean = true,

    // 옵티마이저
    val learningRate: Double = 1e-3,
    val maxIters: Int = 5000,
    val weightDecay: Double = 1e-1,
    val beta1: Double = 0.9,
    val beta2: Double = 0.99,
    val gradClip: Double = 1.0,

    // 학습률 스케줄
    val decayLr: Boolean = true,
    val warmupIters: Int = 100,
    val lrDecayIters: Int = 5000,
    val minLr: Double = 1e-4
)








