package train.experiments

import mps.MpsBackend
import turbo.TurboTrainConfig
import turbo.TurboTrainer

/**
 * Bench10MTurbo와 동일 10M 모델 / 같은 config. `MpsBackend.enable()`로 MatMul만 Metal GPU.
 *
 * 진입점 분리만으로 turbo vs mps 직접 wall-clock 비교 가능 (expName="bench10m-mps").
 * Mps unavailable이면 turbo CPU로 fallback.
 */
fun main(args: Array<String>) {
    val mpsOk = MpsBackend.enable()
    println("[mps] enabled=$mpsOk")

    val maxIters = args.firstOrNull()?.toIntOrNull() ?: 250

    val config = TurboTrainConfig(
        dataPath = "data/ccmc-v2-pro/stage2",
        modelDir = "model",
        expName = "bench10m-mps",
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
