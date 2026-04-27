package train.experiments

import train.TrainConfig
import train.ScalarTrainer

/**
 * TinyHelen **conversation-only** 코퍼스 학습 — 벡터 백엔드 엔트리.
 *
 * 100M leaner의 conversation 카테고리 단독(191 train / 9 val docs, ~831k train tok).
 *
 * **v4**: v3(1M, layers 4·dim 128)에서 best avg 3.30에 도달했으나 샘플 품질이 여전히
 * 표면적 패턴 수준 → **모델 크기 2× 확대**:
 *   - layers 4 → **6**
 *   - dim 128 → **160** (heads 4, head dim 40)
 *   - 파라미터 수: 1,057,536 → **2,186,240** (≈2.07×)
 *
 * Regularization은 v3 전량 유지 (과적합 여전히 우려):
 *   - batch × accum = 2 × 32 = 64
 *   - LR 3e-4 / min 3e-5
 *   - weightDecay 0.1
 *   - labelSmoothing 0.1
 *   - dropout 0.1
 *   - gradClip 2.0
 *   - maxIters 12000 (2× 모델에서 Chinchilla ×1.12 = 토큰당 28.5x)
 *
 * 체크포인트 경로는 파라미터 수 변경으로 v3과 자동 격리:
 *   `model/tinyhelen-conversation/vec/2186240/<lossInt>/`
 *
 * CLI 인자 규약은 동일 — 숫자=maxIters override, `"resume"`=이어하기.
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TrainConfig(
        dataPath = "data/tinyhelen-conversation",
        modelDir = "model",
        // effective batch = 2 * 32 = 64
        gradientAccumulationSteps = 32,
        batchSize = 2,
        blockSize = 64,
        // 모델 v4 (≈2.19M params, v3 1M의 2.07×)
        numberOfLayers = 6,
        numberOfHeads = 4,
        embeddingDimension = 160,
        dropout = 0.1f,
        bias = true,
        // 옵티마이저 — v2 regularization 강화
        learningRate = 3e-4f,
        weightDecay = 0.1f,
        labelSmoothing = 0.1f,
        gradClip = 2.0f,
        beta1 = 0.9f,
        beta2 = 0.95f,
        // 스케줄
        //   - warmup 3% (0.03 * 12000 = 360 iter)
        //   - cosine decay 95%까지 (iter 11400), 마지막 5%는 min LR plateau
        maxIters = maxItersOverride ?: 12000,
        warmupRatio = 0.03f,
        learningRateDecayRatio = 0.95f,
        minimumLearningRate = 3e-5f,
        decayLr = true,
        // 평가/로깅
        //   12000 × 0.05 = 600 iter마다 eval → 20회
        evalIntervalRatio = 0.05f,
        evalIters = 100,
        logInterval = 100,
        alwaysSaveCheckpoint = false,
        initFrom = if (resume) "resume" else "scratch",
    )

    vec.VecTrainer(config).train()
}
