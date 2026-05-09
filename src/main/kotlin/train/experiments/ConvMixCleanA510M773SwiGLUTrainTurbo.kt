package train.experiments

import train.TrainConfig
import train.ScalarTrainer

/**
 * **conv-mix-clean-a510 + 773k tied + SwiGLU MLP** — modern architecture upgrade.
 *
 * 직전 ConvMixCleanA510M773TrainVec(GELU MLP, val 2.87)에서 한 단계 더 — Llama 스타일
 * SwiGLU activation으로 변경. GELU 대비 일관되게 perplexity 2-5% 개선 보고됨 (Touvron 2023).
 *
 * 모델 구조:
 *   - layers 6 · dim 96 · heads 3 · head dim 32 · block 64 · tied weights
 *   - **MLP: SwiGLU**, hidden = round(8/3 × 96) = 256 (vs GELU 4×96=384)
 *   - 3개 Linear (gate, up, down) — params는 GELU와 거의 같음 (~774k 총합)
 *
 * 학습 설정: clean-a510 baseline과 동일 (직접 비교)
 *   - LR 3e-4 / dropout 0.05 / wd 0.05 / LS 0.05 / gradClip 1.0
 *   - maxIters 12000
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TrainConfig(
        dataPath = "data/conv-mix-clean-a510",
        modelDir = "model",
        expName = "m773-swiglu",
        gradientAccumulationSteps = 32,
        batchSize = 2,
        blockSize = 64,
        numberOfLayers = 6,
        numberOfHeads = 3,
        embeddingDimension = 96,
        dropout = 0.05f,
        bias = true,
        tieWeights = true,
        mlpActivation = "swiglu",  // ← 이 entry만 SwiGLU 사용
        learningRate = 3e-4f,
        weightDecay = 0.05f,
        labelSmoothing = 0.05f,
        gradClip = 1.0f,
        beta1 = 0.9f,
        beta2 = 0.95f,
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
