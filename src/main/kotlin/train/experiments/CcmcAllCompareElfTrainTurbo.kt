package train.experiments

import turbo.TurboTrainConfig
import turbo.TurboTrainer

/**
 * ccmc-all-v2048-v7 + WiderH2 485k 설정으로 ELF와 동일 조건 학습 (10000 iter).
 *
 * v7 = ccmc-all corpus + BOS/EOS 래핑 (record 종결 토큰 학습 신호 포함).
 *
 * ELF의 `runElfTrainCompare`와 모델 크기/데이터셋/하이퍼파라미터를 매칭:
 *   embedDim=64, layers=7, heads=2, batchSize=8, blockSize=64, gradAccum=4,
 *   lr=3e-4 (cosine warmup 5% → decay 95%), wd=0.01, beta=(0.9, 0.95), dropout=0.1
 *
 * 기존 `CcmcAllV2048WiderH2TrainTurbo`(main expName)의 장시간 ckpt와 충돌하지 않도록
 * expName="compare-vs-elf"로 분리 저장. 결과 ckpt:
 *   model/ccmc-all-v2048-v7/compare-vs-elf/v0001/
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TurboTrainConfig(
        dataPath = "data/ccmc-all-v2048-v7",
        modelDir = "model",
        expName = "compare-vs-elf",
        gradientAccumulationSteps = 4,
        batchSize = 8,
        blockSize = 64,
        numberOfLayers = 7,
        numberOfHeads = 2,
        embeddingDimension = 64,
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
