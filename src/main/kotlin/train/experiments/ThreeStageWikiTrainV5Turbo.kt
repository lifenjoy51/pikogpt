package train.experiments

import train.TrainConfig

/**
 * **three-stage v5 Stage 2 — wiki pretrain continued (2.93x scale)** — v5 dict ckpt에서 가중치 로드,
 * v4와 동일하게 simplewiki vital articles로 백과 맥락 주입 + dict replay 0.30. architecture는
 * Stage 1과 동일 (emb 144 / layers 9 / heads 6, dropout 0.10).
 *
 * 의도: v4에서 wiki 단계가 의미 매핑을 깜빡이게만 만들었던 한계를 큰 capacity로 돌파.
 *
 * 사용법:
 *   ./gradlew runThreeStageWikiTrainV5Vec --args="<Stage1 v5 dict ckpt 디렉터리>"
 *   ./gradlew runThreeStageWikiTrainV5Vec --args="<dict ckpt> 8000"     # maxIters override
 *   ./gradlew runThreeStageWikiTrainV5Vec --args="resume"
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
            "Stage 1 v5 dict pretrain ckpt 디렉터리가 필요합니다. " +
                "예: ./gradlew runThreeStageWikiTrainV5Vec --args=\"model/dict/vec/<v5 paramCount>/v00XX\""
        }
    }

    val config = TrainConfig(
        dataPath = "data/three-stage-v4/wiki",
        modelDir = "model",
        replayDataPath = "data/three-stage-v4/dict/train.bin",
        replayRatio = 0.30f,
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
