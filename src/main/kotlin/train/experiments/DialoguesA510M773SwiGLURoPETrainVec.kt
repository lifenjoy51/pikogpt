package train.experiments

import train.TrainConfig

/**
 * **dialogues-a510 + 773k tied + SwiGLU + RoPE** — TinyHelen 제거하고 TinyDialogues age-5+10만 사용.
 *
 * conv-mix-clean-a510 베스트(val 2.72)와 동일한 아키텍처로 dialogue-only 코퍼스 효과를 측정.
 *   - 데이터 차이: 191개 TinyHelen 동화체 doc 제거
 *   - 결과 train conversations: 51,229 (clean-a510 기준 동일)
 *   - 결과 val conversations:    9,219
 *
 * 모델: layers 6 · dim 96 · heads 3 · head dim 32 · block 64 · tied + SwiGLU + RoPE (~768k)
 *
 * 학습 설정: clean-a510 베스트와 동일.
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TrainConfig(
        dataPath = "data/dialogues-a510",
        modelDir = "model",
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
        positionEncoding = "rope",
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

    vec.VecTrainer(config).train()
}
