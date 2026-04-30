package train.experiments

import train.TrainConfig

/**
 * **three-stage v4 Stage 3 — conv finetune (multi-replay)** — Stage 2 wiki ckpt에서 가중치 로드,
 * TinyDialogues age-5+age-10 (16.3M tokens)로 dialogue 형식 적응. dict와 wiki를 **별도 replay
 * path**로 동시 등록하여 비율을 정밀 제어 (단순 합본 시 dict가 1:8.7로 묻히는 문제 회피).
 *
 * 기본 비율 (조정 가능):
 *   - conv 70% + dict 15% + wiki 15% — dict/wiki 균형 노출
 *   - conv 70% + dict 11% + wiki 19% — natural cat ×5 oversampling 등가 분포
 *
 * 학습 설정:
 *   - initFrom = "pretrain_weights": Stage 2 가중치 로드 + optimizer state reset.
 *   - replayDataPath  = "data/three-stage-v4/dict/train.bin", replayRatio  = 0.15
 *   - replayDataPath2 = "data/three-stage-v4/wiki/train.bin", replayRatio2 = 0.15
 *   - LR 1e-4 / warmup 5% / cosine 0.95 (v2 IT 패턴).
 *   - maxIters 8000 ≈ 2.0 epoch over conv 16.3M tokens.
 *
 * 산출 ckpt 경로: `model/conv/vec/<paramCount>/v00XX/` (datasetName = "conv")
 *
 * 사용법:
 *   ./gradlew runThreeStageConvTrainV4Vec --args="<Stage2 wiki ckpt 디렉터리>"
 *   ./gradlew runThreeStageConvTrainV4Vec --args="<wiki ckpt> 6000"         # maxIters override
 *   ./gradlew runThreeStageConvTrainV4Vec --args="resume"
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
            "Stage 2 wiki pretrain ckpt 디렉터리가 필요합니다. " +
                "예: ./gradlew runThreeStageConvTrainV4Vec --args=\"model/wiki/vec/<paramCount>/v00XX\""
        }
    }

    val config = TrainConfig(
        dataPath = "data/three-stage-v4/conv",
        modelDir = "model",
        replayDataPath = "data/three-stage-v4/dict/train.bin",
        replayRatio = 0.15f,
        replayDataPath2 = "data/three-stage-v4/wiki/train.bin",
        replayRatio2 = 0.15f,
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
        maxIters = maxItersOverride ?: 8000,
        warmupRatio = 0.05f,
        learningRateDecayRatio = 0.95f,
        minimumLearningRate = 1e-5f,
        decayLr = true,
        evalIntervalRatio = 0.05f,
        evalIters = 200,
        logInterval = 100,
        alwaysSaveCheckpoint = false,
        initFrom = if (resume) "resume" else "pretrain_weights",
    )

    vec.VecTrainer(config).train()
}
