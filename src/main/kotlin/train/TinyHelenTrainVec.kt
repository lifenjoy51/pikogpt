package train

/**
 * TinyHelen 코퍼스 overnight 학습 — 벡터 백엔드 (`vec.Trainer`) 용 엔트리.
 *
 * 스칼라 `TinyHelenTrain`과 같은 데이터(`data/tinyhelen`)를 쓰지만
 * 아키텍처는 **약 1M 파라미터**로 확대:
 *   - numberOfLayers        : 4
 *   - embeddingDimension    : 128
 *   - numberOfHeads         : 4  (head dim = 32)
 *   - MLP hidden (FFN)      : 512 (= 4 * embedDim, `vec.layer.MLP`가 자동 설정)
 *   - blockSize             : 48  (컨텍스트는 유지)
 *
 * 대략적 파라미터 수:
 *   tok_emb(1000×128) + pos_emb(48×128) + 4×block(≈198K) + final_ln + lm_head(128×1000)
 *   = 128K + 6K + 793K + 256 + 128K ≈ **1.05M**
 *
 * 스칼라 autodiff로는 이 규모가 비현실적(≈2-4분/iter × 1500 iter). 벡터 백엔드로는
 * iter 훨씬 짧음 (FloatArray loop 기반). 체크포인트는 `${modelDir}/vec/{params}/{loss*10}/`.
 *
 * CLI 인자로 maxIters override 가능 (`--args="20"` 등으로 smoke).
 */
fun main(args: Array<String>) {
    val maxItersOverride = args.getOrNull(0)?.toIntOrNull()

    val config = TrainConfig(
        dataPath = "data/tinyhelen",
        modelDir = "model",
        // 효과 배치 = batch * accum = 8 (노이즈 vs 속도 균형)
        gradientAccumulationSteps = 4,
        batchSize = 2,
        blockSize = 48,
        // 모델 (≈1M 파라미터)
        numberOfLayers = 4,
        numberOfHeads = 4,
        embeddingDimension = 128,
        dropout = 0.0f,
        bias = true,
        // 옵티마이저 (recipe A 기반)
        learningRate = 3e-4f,
        weightDecay = 0.02f,
        gradClip = 1.0f,
        beta1 = 0.9f,
        beta2 = 0.95f,
        // 스케줄
        maxIters = maxItersOverride ?: 1500,
        warmupRatio = 0.05f,
        learningRateDecayRatio = 0.85f,
        minimumLearningRate = 5e-5f,
        decayLr = true,
        // 평가/로깅
        evalIntervalRatio = 0.05f,
        evalIters = 4,
        logInterval = 10,
        alwaysSaveCheckpoint = true,
    )

    vec.Trainer(config).train()
}
