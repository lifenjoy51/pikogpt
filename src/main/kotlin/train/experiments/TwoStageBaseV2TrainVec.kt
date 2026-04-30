package train.experiments

import train.TrainConfig

/**
 * **two-stage v2 BASE pretrain** — TinyHelen leaner/100M의 wiki+textbook+book+web(4 shards)로
 * 사실 지식 주입. v1(8.48M tokens) 대비 6× 데이터 확장 + vocab 1000→2000.
 *
 * 모델 architecture는 v1과 **동일**하게 유지(C=96, L=6, heads=3) — 데이터/vocab 효과를
 * 모델 변화와 분리해 측정. vocab 2× 증가로 token embed만 +96K → 합 ~864K params.
 *
 * 학습 설정:
 *   - maxIters 18000 (v1 10000) — BASE 토큰 ~50M, effective batch 4096 → 1.5 epoch.
 *     v1은 4 epoch × 8.48M = 40M token 노출 후 plateau. v2 데이터는 6× 다양해 1 epoch+ 필요.
 *   - evalIters 200 (v1 100) — val 분포가 4-corpus 합쳐져 다양해짐.
 *   - 나머지 hyperparams는 v1과 동일 (LR 3e-4, warmup 3%, cosine 0.95, label smoothing 0.05, swiglu, RoPE, tied).
 *
 * 산출 ckpt 경로: `model/base-v2/vec/<paramCount>/v00XX/` (datasetName = "base-v2")
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TrainConfig(
        dataPath = "data/two-stage-v2/base-v2",
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
        maxIters = maxItersOverride ?: 18000,
        warmupRatio = 0.03f,
        learningRateDecayRatio = 0.95f,
        minimumLearningRate = 3e-5f,
        decayLr = true,
        evalIntervalRatio = 0.05f,
        evalIters = 200,
        logInterval = 100,
        alwaysSaveCheckpoint = false,
        initFrom = if (resume) "resume" else "scratch",
    )

    vec.VecTrainer(config).train()
}
