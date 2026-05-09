package train.experiments

import train.TrainConfig

/**
 * v4-merged space-sep BPE corpus로 ~3M 파라미터 모델 학습.
 *
 *   - dataPath: data/ccmc-v4-merged-spacesep-4k (vocab=4000, train 6.27M / val 319K)
 *   - 모델: layers 8 · dim 160 · heads 5 (head_dim 32) · **block 128** · tied + SwiGLU + RoPE
 *     (~3.12M params; 1M 대비 폭만 확장: 96→160, hidden 256→427)
 *   - 학습: 3000 iter × 8192 token/step (block 128 × batch 2 × gradAccum 32) ≈ 24.6M token 노출
 *   - 목적: 1M 모델 대비 capacity 3배로 OOV/문장 자연스러움 개선 검증.
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TrainConfig(
        dataPath = "data/ccmc-v4-merged-spacesep-4k",
        modelDir = "model",
        expName = "3m",
        samplePrompts = listOf(
            "the cat ",
            "the water ",
            "the tree ",
            "to run ",
            "to eat ",
            "a happy ",
            "a big ",
            "above ",
            "quickly ",
            "and ",
        ),
        gradientAccumulationSteps = 32,
        batchSize = 2,
        blockSize = 128,
        numberOfLayers = 8,
        numberOfHeads = 5,
        embeddingDimension = 160,
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
        maxIters = maxItersOverride ?: 3000,
        warmupRatio = 0.03f,
        learningRateDecayRatio = 0.95f,
        minimumLearningRate = 3e-5f,
        decayLr = true,
        evalIntervalRatio = 0.05f,  // 150 iter마다 eval (3000 × 0.05) → 20 evals
        evalIters = 100,             // SEM ~0.03, 클린한 val 곡선
        logInterval = 50,
        alwaysSaveCheckpoint = true,
        recordAwareSampling = false,
        chunkAnchoredSampling = true,
        initFrom = if (resume) "resume" else "scratch",
    )

    turbo.TurboTrainer(config).train()
}
