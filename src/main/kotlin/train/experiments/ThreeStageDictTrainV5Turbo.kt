package train.experiments

import train.TrainConfig

/**
 * **three-stage v5 Stage 1 — dict pretrain (2.93x scale)** — v4와 동일 데이터·LR·iter, architecture만
 * 옵션 C로 확대: emb 96→144, layers 6→9, heads 3→6 (head_dim 24 유지). 약 2.53M params (~2.93x).
 *
 * 의도: v4 wiki에서 관찰된 의미 매핑 깜빡임(Run go/move 3/9 ckpt에서만 정합)이 capacity 부족
 * 신호로 판단됨 → MLP 용량(emb²)과 합성 깊이를 함께 늘려 단어→hypernym 매핑을 안정 인코딩.
 *
 * dropout 0.05→0.10 — 파라미터 2.93x인데 dict 토큰은 1.107M로 동일하므로 overfit 보강.
 *
 * 데이터·iter·LR·labelSmoothing·warmup 등 v4와 동일하여 직접 비교 가능.
 *
 * 산출 ckpt 경로: `model/dict/vec/<paramCount>/v00XX/` — paramCount가 v4(864K)와 다르므로 자동 분리.
 *
 * 사용법:
 *   ./gradlew runThreeStageDictTrainV5Vec
 *   ./gradlew runThreeStageDictTrainV5Vec --args="3500"          # maxIters override
 *   ./gradlew runThreeStageDictTrainV5Vec --args="resume"
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TrainConfig(
        dataPath = "data/three-stage-v4/dict",
        modelDir = "model",
        expName = "v5",
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
