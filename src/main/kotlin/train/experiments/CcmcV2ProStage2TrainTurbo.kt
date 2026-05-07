package train.experiments

import train.TrainConfig
import turbo.TurboTrainer

/**
 * CCMC v2-pro Stage 2 (instruction) — **turbo 백엔드 버전**.
 *
 * vec 백엔드의 CcmcV2ProStage2TrainVec와 동일 config + 동일 모델 (8 layers · 96 emb · 3 heads · block 32 ·
 * SwiGLU + RoPE + tied), 차이는 백엔드만. SIMD MatMul 4× 가속 + ForkJoinPool 기대.
 *
 * 모델 파라미터 순서가 vec와 동일하므로 vec stage1 ckpt(`model/stage1/vec/1087936/v00XX`)의
 * `model_weights.bin`을 그대로 로드 가능 (turbo는 메타 무시, 가중치만 binary 적재).
 *
 * 사용법:
 *   ./gradlew runCcmcV2ProStage2TrainTurbo --args="model/stage1/vec/1087936/v0053"
 *   ./gradlew runCcmcV2ProStage2TrainTurbo --args="model/stage1/vec/1087936/v0053 3000"
 *   ./gradlew runCcmcV2ProStage2TrainTurbo --args="resume"
 *
 * Note: TurboTrainer는 `samplePrompts` / `recordAwareSampling` 옵션을 (Phase 0~5에서) 처리하지 않음 —
 * 일반 DataLoader 사용 + 내장 sample prompts. vec 결과와 약간 차이 가능 (학습 동작 자체는 정상).
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
            "Stage 1 pretrain ckpt 디렉터리가 필요합니다. 예: ./gradlew runCcmcV2ProStage2TrainTurbo " +
                "--args=\"model/stage1/vec/1087936/v0053\""
        }
    }

    val config = TrainConfig(
        dataPath = "data/ccmc-v2-pro/stage2",
        modelDir = "model",
        replayDataPath = "data/ccmc-v2-pro/stage1/train.bin",
        replayRatio = 0.25f,
        pretrainCheckpointDir = pretrainCkptDir,
        gradientAccumulationSteps = 32,
        batchSize = 2,
        blockSize = 32,
        numberOfLayers = 8,
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
        maxIters = maxItersOverride ?: 3000,
        warmupRatio = 0.05f,
        learningRateDecayRatio = 0.95f,
        minimumLearningRate = 1e-5f,
        decayLr = true,
        evalIntervalRatio = 0.05f,
        evalIters = 100,
        logInterval = 100,
        alwaysSaveCheckpoint = false,
        recordAwareSampling = true,
        initFrom = if (resume) "resume" else "pretrain_weights",
    )

    TurboTrainer(config).train()
}
