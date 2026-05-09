package train.experiments

import turbo.TurboTrainConfig

/**
 * **three-stage v4 Stage 1 — dict pretrain** — Simple English Dict + WordNet 병합 자연어 doc으로
 * "정의 패턴" grounding. v3 base가 wiki 본문 안에서만 정의 패턴을 한 번씩 만나는 한계를
 * 보완하기 위해 명시적 사전 entry로 시작. scratch 학습.
 *
 * 모델 architecture는 v2와 동일 (C=96, L=6, heads=3, swiglu, RoPE, tied) — 이후 단계가
 * 같은 weight를 이어받아야 하므로 architecture 고정.
 *
 * 학습 설정:
 *   - corpus tokens ≈ 1.107M (의미별 doc 분리 형식, 13,521 entries × 평균 3.26 docs/entry).
 *     batch×accum×block = 4096 tok/iter.
 *   - maxIters 22000 ≈ 81 epoch — small corpus + hapax 40%라 단어당 충분한 노출 횟수 보장.
 *     plateau 진입 시 best ckpt 자동 동결 → 다음 단계는 plateau 시점 ckpt 이어받음.
 *   - LR 3e-4 / warmup 3% / cosine 0.95 (v2 BASE와 동일).
 *   - alwaysSave=false — best avg(train+val)/2 갱신 시에만 저장.
 *
 * 산출 ckpt 경로: `model/dict/vec/<paramCount>/v00XX/` (datasetName = "dict")
 *
 * 사용법:
 *   ./gradlew runThreeStageDictTrainV4Vec
 *   ./gradlew runThreeStageDictTrainV4Vec --args="3500"          # maxIters override
 *   ./gradlew runThreeStageDictTrainV4Vec --args="resume"
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TurboTrainConfig(
        dataPath = "data/three-stage-v4/dict",
        modelDir = "model",
        expName = "v4",
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
        maxIters = maxItersOverride ?: 22000,
        warmupRatio = 0.03f,
        learningRateDecayRatio = 0.95f,
        minimumLearningRate = 3e-5f,
        decayLr = true,
        evalIntervalRatio = 0.02f,
        evalIters = 100,
        logInterval = 50,
        alwaysSaveCheckpoint = true,
        earlyStopPatience = 10,
        initFrom = if (resume) "resume" else "scratch",
    )

    turbo.TurboTrainer(config).train()
}
