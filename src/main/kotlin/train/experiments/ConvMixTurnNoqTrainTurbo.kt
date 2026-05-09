package train.experiments

import turbo.TurboTrainConfig
import train.ScalarTrainer

/**
 * **conv-mix-turn-noq** 학습 — conv-mix-turn에서 outer 따옴표 제거 버전.
 *
 * `<|turn|>` 토큰이 발화 boundary 역할 하니 따옴표는 redundant. 따옴표 제거 시:
 *   - vocab이 약 1.14% chars 줄어 cleaner BPE merge
 *   - 모델 출력에서 따옴표 안 나옴 → chat 표시 깔끔
 *   - 내부 인용(`X said "Y"`)은 보존
 *
 * 데이터: `data/conv-mix-turn-noq/`
 *   - 각 turn segment의 leading/trailing `"` 제거
 *   - 내부 nested quote(약 3,896 turn)는 그대로
 *
 * 모델: 432k params (conv-mix-turn과 동일, Chinchilla 비율 유지)
 *   - layers 6 · dim 64 · heads 2
 *
 * 하이퍼파라미터: conv-mix-turn 동일 (직접 비교 목적):
 *   - batch × accum = 2 × 32 = 64
 *   - LR 3e-4 / dropout 0.05 / wd 0.05 / LS 0.05 / gradClip 1.0
 *   - maxIters 8000
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TurboTrainConfig(
        dataPath = "data/conv-mix-turn-noq",
        modelDir = "model",
        gradientAccumulationSteps = 32,
        batchSize = 2,
        blockSize = 64,
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
