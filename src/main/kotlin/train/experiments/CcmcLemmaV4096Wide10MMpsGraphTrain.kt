package train.experiments

import gpt.GPTConfig
import mps.MpsGraphTrainConfig
import mps.MpsGraphTrainer
import turbo.TurboModelConfig

/**
 * ccmc-lemma corpus + vocab 4096 BPE 재학습 + ~10.5M params 모델 (mps graph backend).
 *
 * v1024/v2048와 backbone 동일 (embedDim=256, numLayers=12, heads=8). vocab만 4096.
 * embedding table만 늘어남: 2048×256=524k → 4096×256=1.05M (+524k).
 * 전체: backbone 9.49M + emb 1.05M = ~10.53M params.
 *
 * 데이터: data/ccmc-lemma-v4096/ (v1024/v2048와 같은 corpus, vocab 4096 BPE 재인코딩)
 * 진입점은 batch 128, lr 3e-4 동일.
 *
 * 출력: model/ccmc-lemma-v4096/wide-10m-mps/v<NNNN>/
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val dataPath = "data/ccmc-lemma-v4096"
    val vocab = 4096

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
