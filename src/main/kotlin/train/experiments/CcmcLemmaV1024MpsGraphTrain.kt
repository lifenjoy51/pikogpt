package train.experiments

import data.MetaInfo
import gpt.GPTConfig
import kotlinx.serialization.json.Json
import mps.MpsGraphTrainConfig
import mps.MpsGraphTrainer
import turbo.TurboModelConfig
import java.io.File

/**
 * P0.5 — ccmc-lemma-v1024 (vocab≈1024, 짧은 문장 코퍼스) 대상 MPSGraph backend 학습 진입점.
 *
 * CcmcLemmaV1024TrainTurbo와 같은 모델 hyperparam(embedDim=32, numLayers=5, numHeads=2, blockSize=32,
 * tieWeights=true)을 mps step graph로 학습. mps backend는 PoC scope이라 swiglu+rope+layernorm 고정.
 *
 * 호출 예:
 *   ./gradlew runCcmcLemmaV1024MpsGraphTrain                # scratch, maxIters=10000
 *   ./gradlew runCcmcLemmaV1024MpsGraphTrain --args="500"    # maxIters=500
 *   ./gradlew runCcmcLemmaV1024MpsGraphTrain --args="resume" # 최신 ckpt 이어 학습
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
        embeddingDimension = 32,
        numberOfLayers = 5,
        numberOfAttentionHeads = 2,
        maxSequenceLength = 32,
        dropoutProbability = 0.0f,
        useBias = true,
    )
    val modelCfg = TurboModelConfig(
        gpt = gptCfg,
        tieWeights = true,
        mlpActivation = "swiglu",
        positionEncoding = "rope",
        normalizationType = "layernorm",
    )
    val trainCfg = MpsGraphTrainConfig(
        dataPath = dataPath,
        modelDir = "model",
        expName = "mps-graph",
        blockSize = 32,
        learningRate = 3e-4f,
        weightDecay = 0.01f,
        gradClip = 1.0f,
        beta1 = 0.9f,
        beta2 = 0.95f,
        eps = 1e-8f,
        maxIters = maxItersOverride ?: 10000,
        warmupRatio = 0.05f,
        learningRateDecayRatio = 0.95f,
        minimumLearningRate = 1e-5f,
        decayLr = true,
        evalIntervalRatio = 0.05f,
        evalIters = 10,
        logInterval = 100,
        alwaysSaveCheckpoint = true,
        earlyStopPatience = 0,
        initFrom = if (resume) "resume" else "scratch",
    )
    MpsGraphTrainer(trainCfg, modelCfg).train()
}
