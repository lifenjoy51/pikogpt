package train

/**
 * **conv-mix** 학습 — TinyHelen 100M conversation + TinyDialogues age-5 (화자 마커 제거).
 *
 * 데이터: `data/conv-mix/`
 *   - TinyHelen 100M leaner conversation 191 docs (영화·소설 발췌, `"X." "Y."` 형식)
 *   - TinyDialogues age-5 28,188 docs (CDS, `**Speaker**:` 마커 strip 후 따옴표만 남김)
 *   - 두 소스 공통적으로 `"발화1." "발화2."` 형식으로 통일
 *   - train 30MB / val 5MB → 약 8.5M / 1.4M 토큰
 *
 * **모델 0.43M** — Chinchilla 권고에 가깝게 맞춘 크기 (8.5M tok / 0.43M params = 19.8×):
 *   - layers **6** · heads **2** · dim **64** · block 64 → 약 432k params (1M baseline의 0.41×)
 *   - heads=2 → head dim 32 (v3 baseline과 같음, 표현력 일관)
 *   - depth-biased: 6 layers로 좁은 width(64) 보완
 *
 * 하이퍼파라미터 (0.5M 모델용 조정):
 *   - batch × accum = 2 × 32 = 64        (유지 — diverse 데이터에선 그대로)
 *   - LR 3e-4 / min 3e-5                 (유지 — 안정 영역)
 *   - weightDecay 0.05                   (1M의 0.1 → 절반, 작은 모델은 과적합 덜)
 *   - labelSmoothing 0.05                (0.1 → 0.05, capacity 작아 confidence 억제 필요 감소)
 *   - dropout 0.05                       (0.1 → 0.05, 작은 모델은 충분히 regularized)
 *   - gradClip 1.0                       (textbook 값, 작은 모델은 stricter)
 *   - maxIters 8000                      (Chinchilla 3.3× = 토큰당 ~38× 노출)
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TrainConfig(
        dataPath = "data/conv-mix",
        modelDir = "model",
        gradientAccumulationSteps = 32,
        batchSize = 2,
        blockSize = 64,
        // 모델 (0.43M, layers 6 × dim 64 × 2 heads, head dim 32 — Chinchilla 19.8×)
        numberOfLayers = 6,
        numberOfHeads = 2,
        embeddingDimension = 64,
        dropout = 0.05f,
        bias = true,
        // 옵티마이저 (작은 모델용 — regularization 절반)
        learningRate = 3e-4f,
        weightDecay = 0.05f,
        labelSmoothing = 0.05f,
        gradClip = 1.0f,
        beta1 = 0.9f,
        beta2 = 0.95f,
        // 스케줄
        maxIters = maxItersOverride ?: 8000,
        warmupRatio = 0.03f,
        learningRateDecayRatio = 0.95f,
        minimumLearningRate = 3e-5f,
        decayLr = true,
        // 평가/로깅
        evalIntervalRatio = 0.05f,
        evalIters = 100,
        logInterval = 100,
        alwaysSaveCheckpoint = false,
        initFrom = if (resume) "resume" else "scratch",
    )

    vec.Trainer(config).train()
}
