package train.experiments

import turbo.TurboTrainConfig
import turbo.TurboTrainer

/**
 * ccmc-lemma-v1024 **deep narrow** 모델 학습 (~1.06M params, 500k iter, 얼리스탑 patience=20).
 *
 * 414k 모델(d=64, L=7) → 동일 width 유지 + **depth 3× 확장** (L=7 → 20).
 *   embedDim=64, numLayers=20, numHeads=2 (head_dim=32 유지), blockSize=32, vocab=1024
 *   추정 params: 토큰+포지션임베드(67,584) + 20×블록(49,408 × 20 = 988,160) + final(128) = ~1.06M
 *   transformer 비중: ~91%
 *
 * 차이점 vs Mid (414k):
 *   - numberOfLayers: 7 → **20**
 *   - expName: "mid-414k" → "deep-1M"
 *   - 나머지 학습 옵션 동일 (LR 1e-4, gradAccum 8, batch 8, smoothed early stop)
 *
 * 주의: L=20은 매우 깊음. vanishing gradient 가능성 — 학습 불안정하면 LR 조정 필요.
 *
 * 출력: model/ccmc-lemma-v1024/deep-1M/v<NNNN>/
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TurboTrainConfig(
        dataPath = "data/ccmc-lemma-v1024",
        modelDir = "model",
        expName = "deep-1M",
        gradientAccumulationSteps = 8,
        batchSize = 8,
        blockSize = 32,
        numberOfLayers = 20,
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
