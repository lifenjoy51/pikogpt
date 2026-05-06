package train.experiments

import train.TrainConfig
import turbo.TurboTrainer

/**
 * TinyHelen 코퍼스 overnight 학습 — turbo 백엔드 진입점.
 *
 * Phase 0 동등성 검증용. vec 백엔드의 TinyHelenTrainVec와 정확히 같은 하이퍼파라미터로
 * 시작해 동일 시드 + 동일 데이터에서 1k iter loss curve가 일치하는지를 본다.
 *
 * 모델 ~1M 파라미터:
 *   - numberOfLayers 4, embeddingDimension 128, numberOfHeads 4, blockSize 64
 *
 * CLI 인자:
 *   - 숫자: maxIters override (smoke)
 *   - "resume": 최신 체크포인트에서 이어 학습
 *
 * 체크포인트 경로: `${modelDir}/${datasetName}/turbo/${paramCount}/v0001/`
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TrainConfig(
        dataPath = "data/tinyhelen",
        modelDir = "model",
        gradientAccumulationSteps = 16,
        batchSize = 2,
        blockSize = 64,
        numberOfLayers = 4,
        numberOfHeads = 4,
        embeddingDimension = 128,
        dropout = 0.0f,
        bias = true,
        learningRate = 3e-4f,
        weightDecay = 0.02f,
        gradClip = 1.0f,
        beta1 = 0.9f,
        beta2 = 0.95f,
        maxIters = maxItersOverride ?: 10000,
        warmupRatio = 0.03f,
        learningRateDecayRatio = 0.95f,
        minimumLearningRate = 3e-5f,
        decayLr = true,
        evalIntervalRatio = 0.05f,
        evalIters = 16,
        logInterval = 100,
        alwaysSaveCheckpoint = true,
        initFrom = if (resume) "resume" else "scratch",
    )

    TurboTrainer(config).train()
}
