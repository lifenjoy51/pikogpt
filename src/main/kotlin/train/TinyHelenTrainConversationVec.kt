package train

/**
 * TinyHelen **conversation-only** 코퍼스 학습 — 벡터 백엔드 엔트리.
 *
 * 100M leaner의 conversation 카테고리 단독(191 train / 9 val docs, ~831k train tok).
 * 같은 1M 아키텍처(layers 4·dim 128·heads 4·block 64)로 textbook run과 비교.
 *
 * `TinyHelenTrainTextbookVec`와 다르게 대화 데이터 특성에 맞춰 하이퍼파라미터 조정.
 * **v2**: 1차 run에서 step 3600에 이미 gap 1.47의 과적합 관찰 → regularization 강화:
 *   - batch × accum = 2 × 32 = 64  (반복 패턴 noise 평활화)
 *   - LR 3e-4 / min 3e-5           (v1 5e-4 → 3e-4, textbook 값으로 되돌려 공격성 완화)
 *   - weightDecay 0.1              (v1 0.02 → 0.1, overfit 억제의 1차 수단)
 *   - labelSmoothing 0.1           (target onehot → 0.9·onehot + 0.1·uniform, overconfidence 완화)
 *   - gradClip 2.0                 (유지 — 후반 grad-norm 여유)
 *   - maxIters 12000               (Chinchilla ×2.32, overfit 전·후 궤적 관찰)
 *
 * 체크포인트 경로는 `config.dataPath`의 마지막 segment에 의해 자동 격리:
 *   `model/tinyhelen-conversation/vec/1057536/<lossInt>/`
 *
 * CLI 인자 규약은 동일 — 숫자=maxIters override, `"resume"`=이어하기.
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TrainConfig(
        dataPath = "data/tinyhelen-conversation",
        modelDir = "model",
        // effective batch = 2 * 32 = 64
        gradientAccumulationSteps = 32,
        batchSize = 2,
        blockSize = 64,
        // 모델 (≈1M params, textbook과 동일)
        numberOfLayers = 4,
        numberOfHeads = 4,
        embeddingDimension = 128,
        dropout = 0.0f,
        bias = true,
        // 옵티마이저 — v2 regularization 강화
        learningRate = 3e-4f,
        weightDecay = 0.1f,
        labelSmoothing = 0.1f,
        gradClip = 2.0f,
        beta1 = 0.9f,
        beta2 = 0.95f,
        // 스케줄
        //   - warmup 3% (0.03 * 12000 = 360 iter)
        //   - cosine decay 95%까지 (iter 11400), 마지막 5%는 min LR plateau
        maxIters = maxItersOverride ?: 12000,
        warmupRatio = 0.03f,
        learningRateDecayRatio = 0.95f,
        minimumLearningRate = 3e-5f,
        decayLr = true,
        // 평가/로깅
        //   12000 × 0.05 = 600 iter마다 eval → 20회
        evalIntervalRatio = 0.05f,
        evalIters = 100,
        logInterval = 100,
        alwaysSaveCheckpoint = false,
        initFrom = if (resume) "resume" else "scratch",
    )

    vec.Trainer(config).train()
}
