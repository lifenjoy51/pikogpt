package train.experiments

import train.TrainConfig

/**
 * **two-stage IT finetune** — BASE pretrain 가중치에서 시작해 dialogues-a510 + 20% BASE replay로
 * dialogue 형식 적응. topic relevance 강화 가설의 두 번째 단계.
 *
 * 데이터:
 *   - primary: `data/two-stage/it/{train,val}.bin` (dialogues-a510, 공유 vocab으로 재인코딩)
 *   - replay: `data/two-stage/base/train.bin` 시퀀스 단위 Bernoulli(p=0.2)
 *
 * 모델: BASE와 동일 architecture (layers 6 · dim 96 · heads 3 · block 64 · tied + SwiGLU + RoPE).
 *
 * 학습 설정:
 *   - initFrom = "pretrain_weights": BASE 가중치 로드 + optimizer state(timeStep, m, v) reset.
 *   - LR 1e-4 (BASE 1/3): finetune 보수적 출발. cosine decay 0.95, warmup 5% (pretrained weight 보호).
 *   - maxIters 8000 ≈ 1.8 epoch over IT 17.9M tokens.
 *
 * 사용법:
 *   ./gradlew runTwoStageITTrainVec --args="<base ckpt 디렉터리>"
 *   ./gradlew runTwoStageITTrainVec --args="<base ckpt 디렉터리> 6000"          # maxIters 오버라이드
 *   ./gradlew runTwoStageITTrainVec --args="resume"                              # 이전 IT ckpt에서 재개
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
            "BASE pretrain ckpt 디렉터리가 필요합니다. 예: ./gradlew runTwoStageITTrainVec --args=\"model/base/vec/<paramCount>/v00XX\""
        }
    }

    val config = TrainConfig(
        dataPath = "data/two-stage/it",
        modelDir = "model",
        replayDataPath = "data/two-stage/base/train.bin",
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
        evalIters = 100,
        logInterval = 100,
        alwaysSaveCheckpoint = false,
        initFrom = if (resume) "resume" else "pretrain_weights",
    )

    turbo.TurboTrainer(config).train()
}
