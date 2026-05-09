package train.experiments

import turbo.TurboTrainConfig

/**
 * v4-merged space-sep BPE corpus로 1M 파라미터 모델 학습.
 *
 *   - dataPath: data/ccmc-v4-merged-spacesep-4k (vocab=4000, 공백 분리 BPE, train 6.27M / val 319K)
 *   - 모델: layers 8 · dim 96 · heads 3 · **block 128** · tied + SwiGLU + RoPE
 *     (~1.25M params; vocab 2K→4K로 embedding 192K → 384K 증가)
 *   - 학습: 3000 iter × 8192 token/step (block 128 × batch 2 × gradAccum 32) ≈ 24.6M token 노출
 *   - 목적: vocab 두 배 늘려 OOV 감소 + 의미 binding 강화 검증 (resume 가능).
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TurboTrainConfig(
        dataPath = "data/ccmc-v4-merged-spacesep-4k",
        modelDir = "model",
        expName = "quick",
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
