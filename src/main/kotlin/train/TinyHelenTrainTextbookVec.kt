package train

/**
 * TinyHelen **textbook-only** 코퍼스 학습 — 벡터 백엔드 (`vec.Trainer`) 엔트리.
 *
 * 동일한 1M 아키텍처를 TinyHelen 전체 혼합 코퍼스가 아닌 textbook 단독(848 docs, ~549k train tok)
 * 으로 학습해 단일 장르 일관성을 관찰하기 위한 설정. `TinyHelenTrainVec`와 다음만 차이:
 *
 *   - dataPath             : `data/tinyhelen-textbook`
 *   - modelDir             : `model-textbook`   (기존 `model/vec/1057536/` 네임스페이스와 격리)
 *   - maxIters             : 6000               (textbook 549k tok × ~11회 노출)
 *   - alwaysSaveCheckpoint : false              (avg best 갱신 시에만 저장 — 디렉토리 clutter 방지)
 *   - evalIters            : 100                (val 61k 중 12.8k 샘플, noise ↓)
 *
 * 나머지(batch 2 × accum 16, blockSize 64, layers 4, dim 128, heads 4, LR 3e-4, warmup 3%,
 * decay 95%, evalIntervalRatio 0.05) 는 동일. CLI 인자 규약도 동일(숫자=maxIters override,
 * `"resume"`=이어하기).
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TrainConfig(
        dataPath = "data/tinyhelen-textbook",
        modelDir = "model-textbook",
        // 효과 배치 = batch * accum = 32
        gradientAccumulationSteps = 16,
        batchSize = 2,
        blockSize = 64,
        // 모델 (≈1M 파라미터, TinyHelenTrainVec와 동일)
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
        //   - warmup 3% (0.03 * 6000 = 180 iter)
        //   - cosine decay 95%까지 (iter 5700), 마지막 5%는 min LR plateau
        //   - min LR 3e-5
        maxIters = maxItersOverride ?: 6000,
        warmupRatio = 0.03f,
        learningRateDecayRatio = 0.95f,
        minimumLearningRate = 3e-5f,
        decayLr = true,
        // 평가/로깅
        //   6000 iter × 0.05 = 300 iter마다 eval → 20회
        //   evalIters 100: 기존 4·16 대비 eval noise 대폭 축소
        evalIntervalRatio = 0.05f,
        evalIters = 100,
        logInterval = 100,
        alwaysSaveCheckpoint = false,
        initFrom = if (resume) "resume" else "scratch",
    )

    vec.Trainer(config).train()
}
