package train.experiments

import turbo.TurboTrainConfig
import turbo.TurboTrainer

/**
 * ccmc-lemma-v1024 중간 사이즈 장기 학습 (~414k params, 500k iter, 얼리스탑 patience=20).
 *
 * 97k 모델(`CcmcLemmaV1024LongTrainTurbo`)의 val CE 2.65 plateau 돌파 목적.
 * pikogpt `CcmcAllV2048WiderH2TrainTurbo` 구조를 lemma vocab=1024로 환산:
 *   embedDim=64, numLayers=7, numHeads=2 (head_dim=32), blockSize=32, vocab=1024
 *   추정 params: 토큰임베드(67k) + 7×블록(345k) + final(0.1k) = ~414k
 *   transformer 비중: ~84%
 *
 * 차이점 vs LongTrain (97k):
 *   - embeddingDimension: 32 → **64**
 *   - numberOfLayers: 5 → **7**
 *   - expName: "long-500k" → "mid-414k"
 *   - 나머지 학습 옵션 동일 (LR 1e-4, gradAccum 8, batch 8, smoothed early stop)
 *
 * 출력: model/ccmc-lemma-v1024/mid-414k/v<NNNN>/
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TurboTrainConfig(
        dataPath = "data/ccmc-lemma-v1024",
        modelDir = "model",
        expName = "mid-414k",
        gradientAccumulationSteps = 8,
        batchSize = 8,
        blockSize = 32,
        numberOfLayers = 7,
        numberOfHeads = 2,
        embeddingDimension = 64,
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
