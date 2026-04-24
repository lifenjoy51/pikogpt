package vec

import data.MetaInfo
import gpt.GPTConfig
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import train.DataLoader
import train.TrainConfig
import vec.layer.PikoGPT
import vec.ops.crossEntropyBackward
import vec.ops.crossEntropyForward
import java.io.File
import java.nio.ByteBuffer
import kotlin.io.path.Path
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * 벡터 백엔드 학습 루프. 스칼라 `train.Trainer`와 같은 skeleton (warmup/cosine LR,
 * gradient accumulation, 주기적 evaluation, best-loss checkpoint 저장)을 따르지만
 * 내부 연산은 전부 `vec.Tensor` + 명시적 forward/backward.
 *
 * 체크포인트 경로 규약: `${config.modelDir}/vec/${paramCount}/${(bestLoss*10).toInt()}/`.
 * 파일 3개:
 *   - `checkpoint.json`  : [VCheckpointMeta] 직렬화
 *   - `model_weights.bin`: 모든 param의 float32를 big-endian 연속 덤프
 *   - `meta.json`        : 학습 데이터 dir의 vocab 메타 복사
 */
class Trainer(private val config: TrainConfig) {

    private lateinit var model: PikoGPT
    private lateinit var optimizer: AdamW
    private lateinit var trainLoader: DataLoader
    private lateinit var valLoader: DataLoader

    private val datasetSize: String by lazy {
        config.calculateTotalParameters(vocabularySize).toString()
    }
    private val modelPath: String by lazy { "${config.modelDir}/vec/$datasetSize" }
    private val baselineLoss: Double by lazy { ln(vocabularySize.toDouble()) }
    private var bestLoss: Double = 0.0
    private var iterationNumber: Int = 0
    private var vocabularySize: Int = 0

    fun train() {
        println("=== PikoGPT (vec 백엔드) 훈련 시작 ===")
        println("설정: $config")
        vocabularySize = readVocabSize()
        bestLoss = baselineLoss
        println("베이스라인 손실: ${"%.4f".format(baselineLoss)} (vocab=$vocabularySize)")

        Path(modelPath).toFile().mkdirs()

        model = PikoGPT(buildModelConfig())
        println("모델 파라미터 텐서 수: ${model.parameters().size}, 총 스칼라 원소: ${model.parameters().sumOf { it.numel }}")

        trainLoader = DataLoader("${config.dataPath}/train.bin", config.batchSize, config.blockSize)
        valLoader = DataLoader("${config.dataPath}/val.bin", config.batchSize, config.blockSize)

        optimizer = AdamW(
            parameters = model.parameters(),
            learningRate = config.learningRate,
            beta1 = config.beta1,
            beta2 = config.beta2,
            weightDecay = config.weightDecay,
        )

        val startTime = System.currentTimeMillis()
        var runningLoss = 0.0
        val evalInterval = config.evalInterval.coerceAtLeast(1)

        while (iterationNumber <= config.maxIters) {
            optimizer.updateLearningRate(getLearningRate(iterationNumber))

            if (iterationNumber % evalInterval == 0) {
                val (trainLoss, valLoss) = estimateLoss()
                val avg = (trainLoss + valLoss) / 2.0
                println(
                    "스텝 $iterationNumber: " +
                        "훈련 ${formatLoss(trainLoss)} | 검증 ${formatLoss(valLoss)} | 평균 $avg"
                )
                if (avg < bestLoss) {
                    bestLoss = avg
                    saveCheckpoint()
                }
            }

            if (iterationNumber == 0 && config.evalOnly) break

            // 한 optimizer step: accum 만큼 gradient를 누적
            val stepLoss = trainStep()
            if (config.gradClip > 0.0f) clipGradients(config.gradClip)
            optimizer.step()
            optimizer.zeroGrad()

            runningLoss = if (runningLoss == 0.0) stepLoss else 0.9 * runningLoss + 0.1 * stepLoss
            if (iterationNumber % config.logInterval == 0) {
                val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
                println("반복 $iterationNumber: 손실 ${formatLoss(runningLoss)}, elapsed ${"%.0f".format(elapsed)}s.")
            }
            iterationNumber++
        }
        println("\n훈련 완료!")
    }

    /**
     * Gradient accumulation: [config.gradientAccumulationSteps] × [config.batchSize] 개의 시퀀스에 대해
     * cross-entropy 평균 loss의 기울기를 누적한다. upstream 스케일은 `1 / (accum * batch)`.
     */
    private fun trainStep(): Double {
        var totalLoss = 0.0
        val accum = config.gradientAccumulationSteps
        val batch = config.batchSize
        val upstreamGrad = 1.0f / (accum * batch)

        for (microStep in 0 until accum) {
            val (inputs, targets) = trainLoader.getBatch()
            for (b in inputs.indices) {
                val logits = model.forward(inputs[b])
                val ce = crossEntropyForward(logits, targets[b])
                totalLoss += ce.loss.toDouble()
                val gLogits = crossEntropyBackward(logits, targets[b], ce.softmax, upstreamGrad)
                model.backward(gLogits)
            }
        }
        return totalLoss / (accum * batch)
    }

    /** train/val 각각 [evalIters]개의 배치를 순차 forward해 평균 cross-entropy loss를 계산. */
    private fun estimateLoss(): Pair<Double, Double> {
        val trainLosses = (0 until config.evalIters).map {
            val (inputs, targets) = trainLoader.getBatch()
            evaluateBatch(inputs, targets)
        }
        val valLosses = (0 until config.evalIters).map {
            val (inputs, targets) = valLoader.getBatch()
            evaluateBatch(inputs, targets)
        }
        return Pair(trainLosses.average(), valLosses.average())
    }

    private fun evaluateBatch(inputs: Array<IntArray>, targets: Array<IntArray>): Double {
        var sum = 0.0
        for (b in inputs.indices) {
            val logits = model.forward(inputs[b])
            sum += crossEntropyForward(logits, targets[b]).loss.toDouble()
        }
        return sum / inputs.size
    }

    /** L2 norm 기반 gradient clipping. */
    private fun clipGradients(maxNorm: Float) {
        var sumSq = 0.0f
        for (p in model.parameters()) {
            val g = p.grad ?: continue
            for (i in g.indices) sumSq += g[i] * g[i]
        }
        val totalNorm = sqrt(sumSq)
        if (totalNorm > maxNorm) {
            val scale = maxNorm / totalNorm
            for (p in model.parameters()) {
                val g = p.grad ?: continue
                for (i in g.indices) g[i] *= scale
            }
        }
    }

    /** warmup → cosine decay → min LR plateau (train.Trainer의 스케줄 수식 복제). */
    private fun getLearningRate(iter: Int): Float {
        if (!config.decayLr) return config.learningRate
        if (iter < config.warmupIters) {
            return config.learningRate * (iter + 1).toFloat() / config.warmupIters.toFloat()
        }
        if (iter > config.learningRateDecayIterations) return config.minimumLearningRate
        val decayRatio =
            (iter - config.warmupIters).toDouble() / (config.learningRateDecayIterations - config.warmupIters)
        val coefficient = 0.5f * (1.0f + cos(PI * decayRatio).toFloat())
        return config.minimumLearningRate + coefficient * (config.learningRate - config.minimumLearningRate)
    }

    private fun saveCheckpoint() {
        val lossInteger = (bestLoss * 10).toInt()
        val dir = File("$modelPath/$lossInteger")
        dir.mkdirs()

        val meta = VCheckpointMeta(
            iterationNumber = iterationNumber,
            bestValidationLoss = bestLoss,
            modelArgs = buildModelConfig(),
            config = config,
        )
        val json = Json { prettyPrint = true; encodeDefaults = true }
        File(dir, "checkpoint.json").writeText(json.encodeToString(meta))
        saveModelWeights(File(dir, "model_weights.bin"))

        val srcMeta = File("${config.dataPath}/meta.json")
        if (srcMeta.exists()) srcMeta.copyTo(File(dir, "meta.json"), overwrite = true)

        println("체크포인트 저장 완료: ${File(dir, "checkpoint.json").absolutePath}")
    }

    private fun saveModelWeights(file: File) {
        file.outputStream().use { out ->
            for (p in model.parameters()) {
                val buf = ByteBuffer.allocate(p.numel * 4)
                for (i in 0 until p.numel) buf.putFloat(p.data[i])
                out.write(buf.array())
            }
        }
    }

    private fun readVocabSize(): Int {
        val meta = File("${config.dataPath}/meta.json").readText()
        val parser = Json { ignoreUnknownKeys = true }
        return parser.decodeFromString<MetaInfo>(meta).vocabSize
    }

    private fun buildModelConfig(): GPTConfig = GPTConfig(
        maxSequenceLength = config.blockSize,
        vocabularySize = vocabularySize,
        numberOfLayers = config.numberOfLayers,
        numberOfAttentionHeads = config.numberOfHeads,
        embeddingDimension = config.embeddingDimension,
        useBias = config.bias,
        dropoutProbability = config.dropout,  // 벡터 백엔드는 현재 dropout 미구현. 저장만.
    )

    private fun formatLoss(loss: Double): String {
        val pct = maxOf(0.0, (baselineLoss - loss) / baselineLoss * 100)
        return "%.2f (%.1f%%)".format(loss, pct)
    }
}
