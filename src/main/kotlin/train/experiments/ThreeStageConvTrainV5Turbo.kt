package train.experiments

import train.TrainConfig

/**
 * **three-stage v5 Stage 3 — conv finetune (2.93x scale)** — v5 wiki ckpt에서 가중치 로드,
 * dict 15% + wiki 15% multi-replay로 conv 적응. architecture는 Stage 1·2와 동일
 * (emb 144 / layers 9 / heads 6, dropout 0.10).
 *
 * 사용법:
 *   ./gradlew runThreeStageConvTrainV5Vec --args="<Stage2 v5 wiki ckpt 디렉터리>"
 *   ./gradlew runThreeStageConvTrainV5Vec --args="<wiki ckpt> 6000"         # maxIters override
 *   ./gradlew runThreeStageConvTrainV5Vec --args="resume"
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
            "Stage 2 v5 wiki pretrain ckpt 디렉터리가 필요합니다. " +
                "예: ./gradlew runThreeStageConvTrainV5Vec --args=\"model/wiki/vec/<v5 paramCount>/v00XX\""
        }
    }

    val config = TrainConfig(
        dataPath = "data/three-stage-v4/conv",
        modelDir = "model",
        expName = "v5",
        replayDataPath = "data/three-stage-v4/dict/train.bin",
        replayRatio = 0.15f,
        replayDataPath2 = "data/three-stage-v4/wiki/train.bin",
        replayRatio2 = 0.15f,
        pretrainCheckpointDir = pretrainCkptDir,
        gradientAccumulationSteps = 32,
        batchSize = 2,
        blockSize = 64,
        numberOfLayers = 9,
        numberOfHeads = 6,
        embeddingDimension = 144,
        dropout = 0.10f,
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
        maxIters = maxItersOverride ?: 24000,
        warmupRatio = 0.05f,
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
