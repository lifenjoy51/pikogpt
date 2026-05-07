package train.experiments

import train.TrainConfig

/**
 * 5M 파라미터 모델 500 iter 벤치 — vec 백엔드 vs turbo 백엔드 시간 비교용.
 *   - 12 layer × 192 emb × 6 head × headDim 32 × block 32 → 약 5.7M params
 *   - SwiGLU + RoPE + tied (stage2 모델 spec scale up)
 *   - data/ccmc-v2-pro/stage2 (vocab 2000) scratch 학습
 *   - 500 iter, eval every 100, log every 50
 */
fun main(args: Array<String>) {
    val maxIters = args.firstOrNull()?.toIntOrNull() ?: 500

    val config = TrainConfig(
        dataPath = "data/ccmc-v2-pro/stage2",
        modelDir = "model/bench5m-vec",
        gradientAccumulationSteps = 16,
        batchSize = 2,
        blockSize = 32,
        numberOfLayers = 12,
        numberOfHeads = 6,
        embeddingDimension = 192,
        dropout = 0.0f,
        bias = true,
        tieWeights = true,
        mlpActivation = "swiglu",
        positionEncoding = "rope",
        learningRate = 3e-4f,
        weightDecay = 0.01f,
        labelSmoothing = 0.0f,
        gradClip = 1.0f,
        beta1 = 0.9f,
        beta2 = 0.95f,
        maxIters = maxIters,
        warmupRatio = 0.05f,
        learningRateDecayRatio = 0.95f,
        minimumLearningRate = 3e-5f,
        decayLr = true,
        evalIntervalRatio = 0.2f,
        evalIters = 25,
        logInterval = 50,
        alwaysSaveCheckpoint = false,
        initFrom = "scratch",
    )

    vec.VecTrainer(config).train()
}
