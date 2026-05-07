package train.experiments

import train.TrainConfig
import turbo.TurboTrainer

/**
 * 10M 파라미터 모델 250 iter — turbo 백엔드 (Bench10MVec와 동일 config).
 */
fun main(args: Array<String>) {
    val maxIters = args.firstOrNull()?.toIntOrNull() ?: 250

    val config = TrainConfig(
        dataPath = "data/ccmc-v2-pro/stage2",
        modelDir = "model/bench10m-turbo",
        gradientAccumulationSteps = 16,
        batchSize = 2,
        blockSize = 32,
        numberOfLayers = 16,
        numberOfHeads = 8,
        embeddingDimension = 256,
        dropout = 0.0f,
        bias = true,
        tieWeights = true,
        mlpActivation = "swiglu",
        positionEncoding = "rope",
        learningRate = 3e-4f,
        weightDecay = 0.01f,
        labelSmoothing = 0.0f,
        gradClip = 1.0f,
        beta1 = 0.9f,
        beta2 = 0.95f,
        maxIters = maxIters,
        warmupRatio = 0.05f,
        learningRateDecayRatio = 0.95f,
        minimumLearningRate = 3e-5f,
        decayLr = true,
        evalIntervalRatio = 0.4f,
        evalIters = 25,
        logInterval = 50,
        alwaysSaveCheckpoint = false,
        initFrom = "scratch",
    )

    TurboTrainer(config).train()
}
