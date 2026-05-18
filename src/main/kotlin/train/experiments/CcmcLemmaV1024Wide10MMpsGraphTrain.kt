package train.experiments

import gpt.GPTConfig
import mps.MpsGraphTrainConfig
import mps.MpsGraphTrainer
import turbo.TurboModelConfig

/**
 * ccmc-lemma-v1024 데이터셋 + ~10M params 모델 (mps graph backend).
 *
 * Width 확장 모델: embedDim=256, numLayers=12, numHeads=8 (head_dim=32 유지),
 * blockSize=32, vocab=1024.
 *   추정 params: tok+pos(256×1024 + 256×32) ≈ 270k + 12×블록(약 786k×12) ≈ 9.43M + final ≈ ~9.7M
 *
 * 차이점 vs Deep (1M, L=20 narrow):
 *   - numberOfLayers: 20 → 12 (안정성)
 *   - embeddingDimension: 64 → 256 (width 4×)
 *   - numberOfAttentionHeads: 2 → 8 (heads 4×, head_dim=32 유지)
 *   - learningRate: 1e-4 → 3e-4 (10M 권장)
 *   - 나머지 학습 옵션 동일 (batch 8, gradAccum 8, dropout 0.1, warmup 0.05, clip 1.0, earlyStop 20)
 *
 * 출력: model/ccmc-lemma-v1024/wide-10m-mps/v<NNNN>/
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val dataPath = "data/ccmc-lemma-v1024"
    val vocab = 1024

    val gptCfg = GPTConfig(
        vocabularySize = vocab,
        embeddingDimension = 256,
        numberOfLayers = 12,
        numberOfAttentionHeads = 8,
        maxSequenceLength = 32,
        dropoutProbability = 0.1f,
        useBias = true,
    )
    val modelCfg = TurboModelConfig(
        gpt = gptCfg,
        tieWeights = true,
        mlpActivation = "gelu",
        positionEncoding = "learned",
        normalizationType = "layernorm",
    )
    val trainCfg = MpsGraphTrainConfig(
        dataPath = dataPath,
        modelDir = "model",
        expName = "wide-10m-mps",
        batchSize = 128,
        gradientAccumulationSteps = 1,
        blockSize = 32,
        learningRate = 3e-4f,
        weightDecay = 0.01f,
        gradClip = 1.0f,
        beta1 = 0.9f,
        beta2 = 0.95f,
        eps = 1e-8f,
        maxIters = maxItersOverride ?: 500000,
        warmupRatio = 0.05f,
        learningRateDecayRatio = 0.95f,
        minimumLearningRate = 1e-5f,
        decayLr = true,
        evalIntervalRatio = 0.01f,
        evalIters = 10,
        logInterval = 1000,
        alwaysSaveCheckpoint = true,
        earlyStopPatience = 20,
        initFrom = if (resume) "resume" else "scratch",
        dropoutProbability = 0.1f,
    )
    MpsGraphTrainer(trainCfg, modelCfg).train()
}
