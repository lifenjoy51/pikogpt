package train.experiments

import train.TrainConfig
import train.ScalarTrainer

/**
 * **conv-mix-a510 + 773k tied + blockSize 128** — 큰 모델에 긴 context 적용.
 *
 * 직전 ConvMixA510M773TrainVec(block 64, val 2.89)에서 응답이 프롬프트와 topic 무관해
 * 보이는 문제 → multi-turn coherence를 위해 blockSize 64→128로 확장.
 *
 * 이전 ConvMixTurnNoqB128는 432k 베이스(block 64) 효과 없거나 더 나빴음. 가설:
 *   - 작은 모델은 긴 context 활용 capacity 부족했음
 *   - 큰 모델(773k tied)은 긴 context로 turn 간 일관성 학습 가능할 것
 *
 * 모델: ~779k params (773k + position emb 6k 추가)
 *   - layers 6 · dim 96 · heads 3 · head dim 32 · block **128** · tied weights
 *
 * 하이퍼파라미터: 773k baseline과 동일 (직접 비교 목적)
 *   - batch × accum = 2 × 32 = 64
 *   - LR 3e-4 / dropout 0.05 / wd 0.05 / LS 0.05 / gradClip 1.0
 *   - maxIters 8000 (block 128로 per-iter 2.5× 느려져 wall clock 비슷하게 유지)
 *
 * Chinchilla: 18.9M / 779k = 24.3× (이전과 동일)
 * 비용: per-iter 약 2.5× 느림 (attention O(block²)). 8000 iter ~2h 예상.
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TrainConfig(
        dataPath = "data/conv-mix-turn-noq-a510",
        modelDir = "model",
        gradientAccumulationSteps = 32,
        batchSize = 2,
        blockSize = 128,
        // 모델: 6 × 96 × 3 + tied (773k 동일, position emb만 64→128로 확장)
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
        maxIters = maxItersOverride ?: 8000,
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
