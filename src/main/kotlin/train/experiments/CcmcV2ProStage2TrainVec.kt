package train.experiments

import train.TrainConfig

/**
 * **CCMC v2-pro Stage 2 (instruction)** — Stage 1 가중치에서 시작해 Q/A 형식 (turn-marked) 데이터로
 * instruction-style 적응. 같은 lemma 위에서 형태만 바뀌므로 binding 망각 방지가 핵심 → Stage 1 replay 0.25.
 *
 * 데이터:
 *   - primary: `data/ccmc-v2-pro/stage2/{train,val}.bin`
 *     형식: `<|bos|><question><|turn|><answer><|turn|>...<|eos|>` (`Q:`/`A:` 마커 제거, `<|turn|>` 사용)
 *   - replay: `data/ccmc-v2-pro/stage1/train.bin` 시퀀스 단위 Bernoulli(p=0.25)
 *
 * 모델: Stage 1과 동일 architecture (layers 8 · dim 96 · heads 3 · block 64 · tied + SwiGLU + RoPE).
 *
 * 학습 설정:
 *   - initFrom = "pretrain_weights": Stage 1 가중치 로드 + optimizer state(timeStep, m, v) reset.
 *   - LR 1e-4 (Stage 1의 1/3): finetune 보수적 출발. cosine decay 0.95, warmup 5% (pretrained weight 보호).
 *   - maxIters 3000 → 3000 × 4096 × 0.75 primary ≈ 9.2M Stage2 token 노출 ≈ 41 epoch over ~222k tokens.
 *
 * 사용법:
 *   ./gradlew runCcmcV2ProStage2TrainVec --args="<stage1 ckpt 디렉터리>"
 *   ./gradlew runCcmcV2ProStage2TrainVec --args="<stage1 ckpt 디렉터리> 4000"   # maxIters 오버라이드
 *   ./gradlew runCcmcV2ProStage2TrainVec --args="resume"                        # 이전 stage2 ckpt에서 재개
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
            "Stage 1 pretrain ckpt 디렉터리가 필요합니다. 예: ./gradlew runCcmcV2ProStage2TrainVec --args=\"model/stage1/vec/<paramCount>/v00XX\""
        }
    }

    val config = TrainConfig(
        dataPath = "data/ccmc-v2-pro/stage2",
        modelDir = "model",
        samplePrompts = listOf(
            "<|bos|>What is a cat?<|turn|>",
            "<|bos|>What is water?<|turn|>",
            "<|bos|>What is a tree?<|turn|>",
            "<|bos|>What does run mean?<|turn|>",
            "<|bos|>What does eat mean?<|turn|>",
            "<|bos|>What does happy mean?<|turn|>",
            "<|bos|>What does big mean?<|turn|>",
            "<|bos|>What does above mean?<|turn|>",
            "<|bos|>What does quickly mean?<|turn|>",
            "<|bos|>What does and mean?<|turn|>",
        ),
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

    vec.VecTrainer(config).train()
}
