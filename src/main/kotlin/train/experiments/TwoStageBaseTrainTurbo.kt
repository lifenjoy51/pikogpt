package train.experiments

import train.TrainConfig

/**
 * **two-stage BASE pretrain** — TinyHelen leaner/100M의 wiki + textbook으로 사실 지식을 주입.
 *
 * topic relevance 강화의 첫 단계. 같은 엔티티(sky, sun, water…)가 여러 사실 컨텍스트에서
 * 반복 등장하는 wiki/textbook 분포로 의미 표현을 형성한다. narrative(book/conversation/
 * TinyStories)는 사용자 결정에 따라 제외 — 어휘가 흩뿌려져 사실 지식 응축에 비효율.
 *
 * 데이터: `data/two-stage/base` (`train.txt` + `val.txt` + `meta.json` 모두 공유 vocab과 동기화 필요)
 *   - 합본 vocab으로 인코딩한 `train.bin` / `val.bin` 사용 (runEncodeWithExistingMeta로 만든다)
 *
 * 모델: layers 6 · dim 96 · heads 3 · block 64 · tied + SwiGLU + RoPE (~768k)
 *   conv-mix-clean-a510 베스트와 동일 architecture로 비교 가능성 보존.
 *
 * 학습 설정:
 *   - maxIters 10000 → effective batch 4096 token/step × 10k = 41M token 노출 ≈ 4 epoch (10M tokens).
 *   - LR 3e-4 (cosine decay 0.95), warmup 3%, label smoothing 0.05 — clean-a510 베스트와 동일.
 *
 * 산출 ckpt 경로: `model/base/vec/<paramCount>/v00XX/` (datasetName = "base")
 *   IT 단계에서 `pretrainCheckpointDir`로 이 경로를 가리킨다.
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TrainConfig(
        dataPath = "data/two-stage/base",
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
        maxIters = maxItersOverride ?: 10000,
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
