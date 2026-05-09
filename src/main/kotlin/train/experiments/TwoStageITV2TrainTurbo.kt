package train.experiments

import turbo.TurboTrainConfig

/**
 * **two-stage v2 IT finetune** — v2 BASE 가중치에서 시작해 dialogues-a510 + 20% v2 BASE replay로
 * dialogue 형식 적응. v2 BASE의 확장된 사실 표현이 dialogue로 transfer되는지 측정.
 *
 * 모델 architecture는 BASE와 동일 (C=96, L=6, heads=3, vocab 2000) — pretrain weight 호환 필수.
 *
 * 학습 설정:
 *   - initFrom = "pretrain_weights": BASE 가중치 로드 + optimizer state(timeStep, m, v) reset.
 *   - replayDataPath = "data/two-stage-v2/base-v2/train.bin", replayRatio = 0.2.
 *   - LR 1e-4 (BASE 1/3) — finetune 보수적 출발.
 *   - maxIters 8000 ≈ 1.8 epoch over IT 18.3M tokens.
 *
 * 사용법:
 *   ./gradlew runTwoStageITV2TrainVec --args="<v2 base ckpt 디렉터리>"
 *   ./gradlew runTwoStageITV2TrainVec --args="<v2 base ckpt> 6000"           # maxIters override
 *   ./gradlew runTwoStageITV2TrainVec --args="resume"                         # 이전 v2 IT ckpt 재개
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
            "v2 BASE pretrain ckpt 디렉터리가 필요합니다. 예: ./gradlew runTwoStageITV2TrainVec --args=\"model/base-v2/vec/864000/v00XX\""
        }
    }

    val config = TurboTrainConfig(
        dataPath = "data/two-stage-v2/it-v2",
        modelDir = "model",
        replayDataPath = "data/two-stage-v2/base-v2/train.bin",
        replayRatio = 0.2f,
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

    turbo.TurboTrainer(config).train()
}
