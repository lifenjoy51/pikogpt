package train.experiments

import train.TrainConfig
import train.ScalarTrainer

/**
 * **conv-mix-turn** 학습 — TinyHelen conv + TinyDialogues age-5,
 * 발화 사이에 명시적 `<|turn|>` 특수 토큰 삽입.
 *
 * 직전 conv-mix run(별도 화자 마커 없음, 따옴표만)에서 모델이 turn 경계를 학습 못해
 * 응답이 한 발화로 끝나지 않고 책처럼 이어지는 문제 관찰 → turn 토큰으로 해결.
 *
 * 데이터: `data/conv-mix-turn/`
 *   - TinyHelen 100M conv: `"X." "Y."` → `"X."<|turn|>"Y."`
 *   - TinyDialogues age-5: `**Speaker**: "..."` → `<|turn|>"..."`
 *   - 평균 13.8 turns/doc
 *
 * 모델: 432k params (conv-mix와 동일, Chinchilla 비율 유지)
 *   - layers 6 · dim 64 · heads 2 · head dim 32 · block 64
 *
 * 하이퍼파라미터 (conv-mix와 동일):
 *   - batch × accum = 2 × 32 = 64
 *   - LR 3e-4 / min 3e-5
 *   - weightDecay 0.05 · labelSmoothing 0.05 · dropout 0.05
 *   - gradClip 1.0 · maxIters 8000
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TrainConfig(
        dataPath = "data/conv-mix-turn",
        modelDir = "model",
        gradientAccumulationSteps = 32,
        batchSize = 2,
        blockSize = 64,
        // 모델 (432k, conv-mix와 동일)
        numberOfLayers = 6,
        numberOfHeads = 2,
        embeddingDimension = 64,
        dropout = 0.05f,
        bias = true,
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
        // 평가/로깅
        evalIntervalRatio = 0.05f,
        evalIters = 100,
        logInterval = 100,
        alwaysSaveCheckpoint = false,
        initFrom = if (resume) "resume" else "scratch",
    )

    turbo.TurboTrainer(config).train()
}
