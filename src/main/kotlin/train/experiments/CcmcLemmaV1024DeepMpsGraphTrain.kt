package train.experiments

import data.MetaInfo
import gpt.GPTConfig
import kotlinx.serialization.json.Json
import mps.MpsGraphTrainConfig
import mps.MpsGraphTrainer
import turbo.TurboModelConfig
import java.io.File

/**
 * ccmc-lemma-v1024 deep-1M (~1.06M params) MPSGraph backend 진입점.
 *
 * CcmcLemmaV1024DeepTrainTurbo와 100% 동일한 hyperparam (P4 후):
 *   embedDim=64, numLayers=20, numHeads=2, blockSize=32, vocab=1024
 *   batchSize=8, gradientAccumulationSteps=8 (effective batch=64)
 *   LR=1e-4, weightDecay=0.01, gradClip=1.0, beta1=0.9, beta2=0.95
 *   maxIters=500000, warmupRatio=0.05, learningRateDecayRatio=0.95, minLR=1e-5
 *   evalIntervalRatio=0.01, evalIters=10, logInterval=1000
 *   alwaysSaveCheckpoint=true, earlyStopPatience=20, tieWeights=true
 *   mlpActivation="gelu", positionEncoding="learned", dropoutProbability=0.1
 *
 * 출력: model/ccmc-lemma-v1024/deep-1M-mps/v<NNNN>/
 *
 * 호출 예:
 *   ./gradlew runCcmcLemmaV1024DeepMpsGraphTrain
 *   ./gradlew runCcmcLemmaV1024DeepMpsGraphTrain --args="500"
 *   ./gradlew runCcmcLemmaV1024DeepMpsGraphTrain --args="resume"
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val dataPath = "data/ccmc-lemma-v1024"
    val metaText = File(dataPath, "meta.json").readText()
    val vocab = Json { ignoreUnknownKeys = true }
        .decodeFromString<MetaInfo>(metaText).vocabularySize

    val gptCfg = GPTConfig(
        vocabularySize = vocab,
        embeddingDimension = 64,
        numberOfLayers = 20,
        numberOfAttentionHeads = 2,
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
        expName = "deep-1M-mps",
        batchSize = 8,
        gradientAccumulationSteps = 8,
        blockSize = 32,
        learningRate = 1e-4f,
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
