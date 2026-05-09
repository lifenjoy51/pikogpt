package train.experiments

import train.TrainConfig
import train.ScalarTrainer

/**
 * **conv-mix-a510 + 773k 모델 (tied weights)** — 데이터 18.9M에 맞춘 모델 크기 키움.
 *
 * 직전 a510 432k untied는 Chinchilla 41×로 데이터 풍부하나 capacity 부족 신호.
 * 모델 크기 ~1.8× 키워서 Chinchilla 권고 비율(20×)에 근접.
 *
 * 모델: 773k params
 *   - layers 6 · dim 96 · heads 3 · head dim 32 · block 64 · tied weights
 *   - depth 우세 (6 layers) + dim 96 (432k의 64 대비 1.5×)
 *   - tied로 임베딩 dual-use 학습 신호 강화 + vocab×dim 절약
 *   - Chinchilla 24.4× (18.9M / 773k) — 권고(20×) 약간 초과로 학습 안정
 *
 * 하이퍼파라미터: 432k baseline 그대로 (직접 비교 목적)
 *   - batch × accum = 2 × 32 = 64
 *   - LR 3e-4 / dropout 0.05 / wd 0.05 / LS 0.05 / gradClip 1.0
 *   - maxIters 12000 (큰 모델은 더 많은 step 필요 — 432k의 8000보다 늘림)
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TrainConfig(
        dataPath = "data/conv-mix-turn-noq-a510",
        modelDir = "model",
        gradientAccumulationSteps = 32,
        batchSize = 2,
        blockSize = 64,
        // 모델: 6 × 96 × 3 + tied = 773k
        numberOfLayers = 6,
        numberOfHeads = 3,
        embeddingDimension = 96,
        dropout = 0.05f,
        bias = true,
        tieWeights = true,
        // 옵티마이저
        learningRate = 3e-4f,
        weightDecay = 0.05f,
        labelSmoothing = 0.05f,
        gradClip = 1.0f,
        beta1 = 0.9f,
        beta2 = 0.95f,
        // 스케줄
        maxIters = maxItersOverride ?: 12000,
        warmupRatio = 0.03f,
        learningRateDecayRatio = 0.95f,
        minimumLearningRate = 3e-5f,
        decayLr = true,
        evalIntervalRatio = 0.05f,
        evalIters = 100,
        logInterval = 100,
        alwaysSaveCheckpoint = false,
        initFrom = if (resume) "resume" else "scratch",
    )

    turbo.TurboTrainer(config).train()
}
