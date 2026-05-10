package train.experiments

import turbo.TurboTrainConfig
import turbo.TurboTrainer

/**
 * v5-qa Stage 2 dialogues로 3M 모델 IT(instruction tuning) — turbo 백엔드.
 *
 * Base: `model/ccmc-v4-merged-spacesep-4k/vec/3117552/v0022` (val 1.82, 3.12M params)
 * 모델 spec은 v0022와 동일 (8 layers · 160 emb · 5 heads · block 128 · SwiGLU + RoPE + tied + bias).
 * vec 학습 ckpt의 `model_weights.bin`을 turbo가 그대로 로드 가능 (params 순서 동일).
 *
 * Replay: base train.bin(`data/ccmc-v4-merged-spacesep-4k/train.bin`)을 25% 섞어
 * narrative 능력 보존 + QA 능력 강화 (catastrophic forgetting 완화).
 *
 * 사용법:
 *   ./gradlew runCcmcV5QaItTrainTurbo --args="model/ccmc-v4-merged-spacesep-4k/vec/3117552/v0022"
 *   ./gradlew runCcmcV5QaItTrainTurbo --args="model/ccmc-v4-merged-spacesep-4k/vec/3117552/v0022 5000"
 *   ./gradlew runCcmcV5QaItTrainTurbo --args="resume"
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
            "base ckpt 디렉터리가 필요합니다. 예: ./gradlew runCcmcV5QaItTrainTurbo " +
                "--args=\"model/ccmc-v4-merged-spacesep-4k/vec/3117552/v0022\""
        }
    }

    val config = TurboTrainConfig(
        dataPath = "data/ccmc-v5-qa-4k",
        modelDir = "model",
        expName = "qa-it",
        replayDataPath = "data/ccmc-v4-merged-spacesep-4k/train.bin",
        replayRatio = 0.25f,
        pretrainCheckpointDir = pretrainCkptDir,
        gradientAccumulationSteps = 32,
        batchSize = 2,
        blockSize = 128,
        numberOfLayers = 8,
        numberOfHeads = 5,
        embeddingDimension = 160,
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
        maxIters = maxItersOverride ?: 3000,
        warmupRatio = 0.05f,
        learningRateDecayRatio = 0.95f,
        minimumLearningRate = 1e-5f,
        decayLr = true,
        evalIntervalRatio = 0.05f,   // 150 iter마다 eval (3000 × 0.05) → 20 evals
        evalIters = 100,
        logInterval = 50,
        alwaysSaveCheckpoint = true,
        recordAwareSampling = true,  // dialogue 한 record 안에서 sampling — Q/A pair 보존
        initFrom = if (resume) "resume" else "pretrain_weights",
    )

    TurboTrainer(config).train()
}
