package train.experiments

import gpt.GPTConfig
import mps.MpsGraphTrainConfig
import mps.MpsGraphTrainer
import turbo.TurboModelConfig

/**
 * ccmc-lemma corpus + vocab 2048 BPE 재학습 + ~10M params 모델 (mps graph backend).
 *
 * CcmcLemmaV1024Wide10MMpsGraphTrain의 vocab만 1024 → 2048 변경, 나머지 hyperparam 동일.
 * Width 확장 모델: embedDim=256, numLayers=12, numHeads=8 (head_dim=32), blockSize=32, vocab=2048.
 *   추정 params: tok+pos(2048×256 + 32×256) ≈ 532k + 12×블록(약 786k×12) ≈ 9.43M = ~9.97M
 *
 * 데이터: data/ccmc-lemma-v2048/ (v1024와 같은 corpus, vocab 2배로 재 BPE 인코딩)
 * 진입점은 batch 128, lr 3e-4 (10M wide 마지막 설정 그대로).
 *
 * 출력: model/ccmc-lemma-v2048/wide-10m-mps/v<NNNN>/
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val dataPath = "data/ccmc-lemma-v2048"
    val vocab = 2048

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
