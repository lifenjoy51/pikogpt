package train.experiments

import turbo.TurboTrainConfig
import train.ScalarTrainer

/**
 * **conv-mix-turn-noq-b128** — conv-mix-turn-noq에서 blockSize만 64→128로 확장.
 *
 * 같은 데이터(`data/conv-mix-turn-noq`), 같은 모델(432k base + position emb 약간 늘어 ~436k),
 * 같은 regularization. 차이는 attention context 길이만:
 *   - blockSize 64 → 128: 한 forward에서 여러 turn을 동시에 볼 수 있음 → long-range coherence ↑
 *   - per-iter 토큰 2× (4096 → 8192) → maxIters 8000 → 4000으로 줄여 같은 노출 유지
 *   - attention O(block²) → 4× 느려지나 모델 작아 감당 가능
 *
 * 모델: 432,128 + 64×64 = 436,224 params (position emb 4k 추가만).
 *
 * 하이퍼파라미터 (이전과 동일):
 *   - batch × accum = 2 × 32 = 64
 *   - LR 3e-4 / dropout 0.05 / wd 0.05 / LS 0.05 / gradClip 1.0
 *   - **maxIters 4000** (Chinchilla 22× 유지)
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TurboTrainConfig(
        dataPath = "data/conv-mix-turn-noq",
        modelDir = "model",
        expName = "b128",
        gradientAccumulationSteps = 32,
        batchSize = 2,
        blockSize = 128,
        numberOfLayers = 6,
        numberOfHeads = 2,
        embeddingDimension = 64,
        dropout = 0.05f,
        bias = true,
        learningRate = 3e-4f,
        weightDecay = 0.05f,
        labelSmoothing = 0.05f,
        gradClip = 1.0f,
        beta1 = 0.9f,
        beta2 = 0.95f,
        maxIters = maxItersOverride ?: 4000,
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
