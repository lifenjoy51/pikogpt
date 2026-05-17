package train.experiments

import mps.MpsBackend
import turbo.TurboTrainConfig
import turbo.TurboTrainer

/**
 * ccmc-lemma-v1024 MPS(Metal GPU) 학습 진입점.
 *
 * [CcmcLemmaV1024TrainTurbo]와 동일한 97k 모델 / 10k iter / config. 차이는
 * 학습 시작 전에 [MpsBackend.enable]을 호출해 `matmulImpl`을 Metal로 교체하는 것뿐.
 * Mps unavailable 시(미macOS, dylib 미빌드, init 실패)는 turbo CPU로 자동 fallback.
 *
 *   expName="mps" — checkpoint는 model/ccmc-lemma-v1024/mps/v0001/ 에 쓰임.
 */
fun main(args: Array<String>) {
    val mpsOk = MpsBackend.enable()
    println("[mps] enabled=$mpsOk")

    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TurboTrainConfig(
        dataPath = "data/ccmc-lemma-v1024",
        modelDir = "model",
        expName = "mps",
        gradientAccumulationSteps = 4,
        batchSize = 8,
        blockSize = 32,
        numberOfLayers = 5,
        numberOfHeads = 2,
        embeddingDimension = 32,
        dropout = 0.1f,
        bias = true,
        learningRate = 3e-4f,
        weightDecay = 0.01f,
        gradClip = 1.0f,
        beta1 = 0.9f,
        beta2 = 0.95f,
        maxIters = maxItersOverride ?: 10000,
        warmupRatio = 0.05f,
        learningRateDecayRatio = 0.95f,
        minimumLearningRate = 1e-5f,
        decayLr = true,
        evalIntervalRatio = 0.05f,
        evalIters = 10,
        logInterval = 100,
        alwaysSaveCheckpoint = true,
        earlyStopPatience = 0,
        initFrom = if (resume) "resume" else "scratch",
        modelCheckpointDir = null,
        tieWeights = true,
    )

    TurboTrainer(config).train()
}
