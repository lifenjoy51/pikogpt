package train.experiments

import turbo.TurboTrainConfig

/**
 * **three-stage v4 Stage 2 — wiki pretrain (continued)** — Stage 1 dict ckpt에서 가중치 로드,
 * simplewiki vital articles L1-L4 본문(8,048 docs / 7.5M tokens, 1 line=1 doc 변환됨)으로 백과
 * 맥락 지식 주입. dict replay 0.30으로 정의 패턴 잊지 않도록.
 *
 * 학습 설정:
 *   - initFrom = "pretrain_weights": Stage 1 가중치 로드 + optimizer state(timeStep, m, v) reset.
 *   - replayDataPath = "data/three-stage-v4/dict/train.bin", replayRatio = 0.30.
 *   - LR 1e-4 (Stage 1의 1/3) — 사전학습된 weight 보호하며 도메인 적응.
 *   - maxIters 20000 ≈ 10.9 epoch over wiki 7.5M tokens (4096 tok/iter).
 *     hapax 단어가 ~11회 노출돼야 사실 학습이 정착 + plateau 진입 보장.
 *   - warmup 1.5% — pretrain 가중치 보존 위해 짧게.
 *
 * 산출 ckpt 경로: `model/wiki/vec/<paramCount>/v00XX/` (datasetName = "wiki")
 *
 * 사용법:
 *   ./gradlew runThreeStageWikiTrainV4Vec --args="<Stage1 dict ckpt 디렉터리>"
 *   ./gradlew runThreeStageWikiTrainV4Vec --args="<dict ckpt> 8000"     # maxIters override
 *   ./gradlew runThreeStageWikiTrainV4Vec --args="resume"
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val nonResumeArgs = args.filter { !it.equals("resume", ignoreCase = true) }

    val pretrainCkptDir: String? = if (resume) null else nonResumeArgs.firstOrNull {
        it.toIntOrNull() == null
    }
    val maxItersOverride: Int? = nonResumeArgs.firstNotNullOfOrNull { it.toIntOrNull() }

    if (!resume) {
        require(pretrainCkptDir != null) {
            "Stage 1 dict pretrain ckpt 디렉터리가 필요합니다. " +
                "예: ./gradlew runThreeStageWikiTrainV4Vec --args=\"model/dict/vec/<paramCount>/v00XX\""
        }
    }

    val config = TurboTrainConfig(
        dataPath = "data/three-stage-v4/wiki",
        modelDir = "model",
        expName = "v4",
        replayDataPath = "data/three-stage-v4/dict/train.bin",
        replayRatio = 0.30f,
        pretrainCheckpointDir = pretrainCkptDir,
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
        learningRate = 1e-4f,
        weightDecay = 0.05f,
        labelSmoothing = 0.05f,
        gradClip = 1.0f,
        beta1 = 0.9f,
        beta2 = 0.95f,
        maxIters = maxItersOverride ?: 20000,
        warmupRatio = 0.015f,
        learningRateDecayRatio = 0.95f,
        minimumLearningRate = 1e-5f,
        decayLr = true,
        evalIntervalRatio = 0.02f,
        evalIters = 200,
        logInterval = 100,
        alwaysSaveCheckpoint = true,
        earlyStopPatience = 10,
        initFrom = if (resume) "resume" else "pretrain_weights",
    )

    turbo.TurboTrainer(config).train()
}
