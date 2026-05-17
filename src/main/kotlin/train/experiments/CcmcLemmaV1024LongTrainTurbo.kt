package train.experiments

import turbo.TurboTrainConfig
import turbo.TurboTrainer

/**
 * ccmc-lemma-v1024 장기 학습 (500k iter, 얼리스탑 patience=20).
 *
 * 모델은 `CcmcLemmaV1024TrainTurbo`와 동일 (~97k params):
 *   embedDim=32, numLayers=5, numHeads=2, blockSize=32, maxSeqLen=32, vocab=1024
 *
 * 차이점:
 *   - maxIters: 10000 → **500000**
 *   - evalIntervalRatio: 0.05 → **0.01** (5000 iter마다 eval, 총 100 ckpt)
 *   - earlyStopPatience: 0 → **20** (20번 = 100k iter no improvement면 종료)
 *   - expName: "main" → "long-500k" (기존 ckpt 디렉터리와 분리)
 *
 * 출력: model/ccmc-lemma-v1024/long-500k/v<NNNN>/
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TurboTrainConfig(
        dataPath = "data/ccmc-lemma-v1024",
        modelDir = "model",
        expName = "long-500k",
        gradientAccumulationSteps = 8,
        batchSize = 8,
        blockSize = 32,
        numberOfLayers = 5,
        numberOfHeads = 2,
        embeddingDimension = 32,
        dropout = 0.1f,
        bias = true,
        learningRate = 1e-4f,
        weightDecay = 0.01f,
        gradClip = 1.0f,
        beta1 = 0.9f,
        beta2 = 0.95f,
        maxIters = maxItersOverride ?: 500000,
        warmupRatio = 0.05f,
        learningRateDecayRatio = 0.95f,
        minimumLearningRate = 1e-5f,
        decayLr = true,
        evalIntervalRatio = 0.01f,
        evalIters = 10,
        logInterval = 1000,
        alwaysSaveCheckpoint = true,
        earlyStopPatience = 20,
        initFrom = if (resume) "resume" else "scratch",
        modelCheckpointDir = null,
        tieWeights = true,
    )

    TurboTrainer(config).train()
}
