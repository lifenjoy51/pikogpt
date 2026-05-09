package train.experiments

import train.TrainConfig
import train.ScalarTrainer

/**
 * **conv-mix-clean-a510 + 773k tied + SwiGLU + RoPE** — modern transformer 변경 둘 다 적용.
 *
 * 직전 SwiGLU run(val 2.81)에서 RoPE position encoding까지 추가:
 *   - Learned position embedding 제거 → token_emb만 사용 (params -6,144)
 *   - Q와 K projection 직후 위치별 회전 적용 → relative position 정보를 attention에 직접 주입
 *   - Long context 일반화 우수 (블록 길이 변경 시 재학습 불필요)
 *
 * 모델: layers 6 · dim 96 · heads 3 · head dim 32 · block 64 · tied + SwiGLU + RoPE
 *   - 파라미터 수: 774k - 6k(pos emb) = ~768k
 *
 * 비교:
 *   - GELU baseline:    val 2.87, perplexity 17.6
 *   - SwiGLU only:      val 2.81, perplexity 16.6
 *   - SwiGLU + RoPE:    이번 실험
 *
 * 학습 설정: clean-a510 baseline과 동일.
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TrainConfig(
        dataPath = "data/conv-mix-clean-a510",
        modelDir = "model",
        expName = "m773-swiglu-rope",
        gradientAccumulationSteps = 32,
        batchSize = 2,
        blockSize = 64,
        numberOfLayers = 6,
        numberOfHeads = 3,
        embeddingDimension = 96,
        dropout = 0.05f,
        bias = true,
        tieWeights = true,
        mlpActivation = "swiglu",
        positionEncoding = "rope",   // ← RoPE 활성
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
