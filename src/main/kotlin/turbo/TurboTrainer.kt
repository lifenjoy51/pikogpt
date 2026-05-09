package turbo

import data.MetaInfo
import data.CharBPE
import gpt.GPTConfig
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import sample.SampleConfig
import train.BatchSource
import train.DataLoader
import turbo.layer.TurboPikoGPT
import turbo.ops.turboCrossEntropyBackward
import turbo.ops.turboCrossEntropyForward
import java.io.File
import java.nio.ByteBuffer
import kotlin.io.path.Path
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * turbo 백엔드 학습 루프. Phase 0은 vec.VecTrainer와 동등 (LR schedule, gradient
 * accumulation, eval, checkpoint, 데이터 병렬 worker-replica).
 *
 * 체크포인트 경로: `${config.modelDir}/${datasetName}/${config.expName}/v0001/`.
 * 같은 datasetName을 공유하는 진입점은 `expName`으로 분리.
 *
 * Phase 2에서 풀링/SIMD, Phase 5에서 worker-replica 폐기 후 ForkJoinPool로 재설계.
 */
class TurboTrainer(
    private val config: TurboTrainConfig,
    /** turbo 전용 알고리즘 옵션. Default OFF면 Phase 0 (vec 동등) 경로 그대로. */
    private val normalizationType: String = "layernorm",
    private val numKvHeads: Int? = null,
    private val useQkNorm: Boolean = false,
    private val useFusedQkv: Boolean = false,
    private val zLossWeight: Float = 0.0f,
    /** Phase 4.4: 모든 block에 gradient checkpointing 적용. dropout=0 require. */
    private val useGradientCheckpointing: Boolean = false,
) {

    init {
        if (useGradientCheckpointing) {
            require(config.dropout == 0.0f) {
                "gradient checkpointing requires dropout=0 (got ${config.dropout})"
            }
        }
    }

    private lateinit var model: TurboPikoGPT
    private lateinit var optimizer: TurboAdamW
    private lateinit var trainLoader: BatchSource
    private lateinit var valLoader: BatchSource
    private lateinit var turboModelConfig: TurboModelConfig

    private lateinit var workers: List<TurboPikoGPT>

    private val datasetName: String by lazy {
        config.dataPath.trimEnd('/').substringAfterLast('/')
    }

    private lateinit var modelPath: String
    private val baselineLoss: Double by lazy { ln(vocabularySize.toDouble()) }
    private var bestLoss: Double = 0.0
    private var iterationNumber: Int = 0
    private var vocabularySize: Int = 0

    private var lastGradNorm: Float = 0.0f
    private var earlyStopCounter: Int = 0

    fun train() {
        println("=== TurboPikoGPT (turbo 백엔드) 훈련 시작 ===")
        println("설정: $config")
        vocabularySize = readVocabSize()
        bestLoss = baselineLoss
        println("베이스라인 손실: ${"%.4f".format(baselineLoss)} (vocab=$vocabularySize)")

        turboModelConfig = buildModelConfig()
        model = TurboPikoGPT(turboModelConfig)
        if (useGradientCheckpointing) {
            for (block in model.blocks) block.useGradientCheckpointing = true
        }
        val actualParamCount = model.parameters().sumOf { it.numel }
        println("모델 파라미터 텐서 수: ${model.parameters().size}, 총 스칼라 원소: $actualParamCount")
        modelPath = "${config.modelDir}/$datasetName/${config.expName}"
        Path(modelPath).toFile().mkdirs()

        val totalSeqsPerIter = config.batchSize * config.gradientAccumulationSteps
        val cpuCount = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)
        val envCap = System.getenv("TURBO_MAX_WORKERS")?.toIntOrNull()?.coerceAtLeast(1)
        // CPU core × 2/3로 default (JIT/GC/OS 및 다른 프로세스 여유 확보)
        val defaultCap = (cpuCount * 2 / 3).coerceAtLeast(1)
        val desiredWorkers = minOf(envCap ?: defaultCap, cpuCount, totalSeqsPerIter)
        workers = if (desiredWorkers >= 2) {
            println(
                "데이터 병렬 학습 활성 — worker 수 $desiredWorkers " +
                    "(CPU $cpuCount, seq/iter $totalSeqsPerIter" +
                    (if (envCap != null) ", TURBO_MAX_WORKERS=$envCap" else "") +
                    ")"
            )
            List(desiredWorkers) {
                TurboPikoGPT(turboModelConfig).also { worker ->
                    if (useGradientCheckpointing) {
                        for (block in worker.blocks) block.useGradientCheckpointing = true
                    }
                }
            }
        } else {
            println("데이터 병렬 비활성 (worker=$desiredWorkers). 순차 경로 사용.")
            emptyList()
        }

        val hasReplay1 = config.replayDataPath != null && config.replayRatio > 0.0f
        val hasReplay2 = config.replayDataPath2 != null && config.replayRatio2 > 0.0f
        trainLoader = when {
            hasReplay1 && hasReplay2 -> {
                println(
                    "Triple train loader 활성: primary=${config.dataPath}/train.bin, " +
                        "replay1=${config.replayDataPath} (ratio=${config.replayRatio}), " +
                        "replay2=${config.replayDataPath2} (ratio=${config.replayRatio2})"
                )
                TripleDataLoader(
                    primaryPath = "${config.dataPath}/train.bin",
                    replay1Path = config.replayDataPath!!,
                    replay2Path = config.replayDataPath2!!,
                    replay1Ratio = config.replayRatio,
                    replay2Ratio = config.replayRatio2,
                    batchSize = config.batchSize,
                    blockSize = config.blockSize,
                )
            }
            hasReplay1 -> {
                println("Mixed train loader 활성: primary=${config.dataPath}/train.bin, replay=${config.replayDataPath}, ratio=${config.replayRatio}")
                MixedDataLoader(
                    primaryPath = "${config.dataPath}/train.bin",
                    replayPath = config.replayDataPath!!,
                    replayRatio = config.replayRatio,
                    batchSize = config.batchSize,
                    blockSize = config.blockSize,
                )
            }
            config.recordAwareSampling -> {
                val bosId = readBosTokenId()
                println("Record-aware train loader 활성: data=${config.dataPath}/train.bin, bosId=$bosId")
                RecordAwareDataLoader(
                    dataPath = "${config.dataPath}/train.bin",
                    batchSize = config.batchSize,
                    blockSize = config.blockSize,
                    bosId = bosId,
                )
            }
            else -> DataLoader("${config.dataPath}/train.bin", config.batchSize, config.blockSize)
        }
        valLoader = if (config.recordAwareSampling) {
            val bosId = readBosTokenId()
            println("Record-aware val loader 활성: data=${config.dataPath}/val.bin, bosId=$bosId")
            RecordAwareDataLoader(
                dataPath = "${config.dataPath}/val.bin",
                batchSize = config.batchSize,
                blockSize = config.blockSize,
                bosId = bosId,
            )
        } else {
            DataLoader("${config.dataPath}/val.bin", config.batchSize, config.blockSize)
        }

        optimizer = TurboAdamW(
            parameters = model.parameters(),
            learningRate = config.learningRate,
            beta1 = config.beta1,
            beta2 = config.beta2,
            weightDecay = config.weightDecay,
        )

        when (config.initFrom) {
            "scratch" -> { /* 신규 학습 */ }
            "resume" -> {
                val latest = findLatestCheckpoint()
                if (latest != null) {
                    loadCheckpoint(latest)
                } else {
                    println("initFrom=resume 이지만 체크포인트를 찾지 못함 — scratch로 진행합니다 (modelPath=$modelPath)")
                }
            }
            "pretrain_weights" -> {
                val src = config.pretrainCheckpointDir
                    ?: error("initFrom=pretrain_weights 일 때 pretrainCheckpointDir이 필요합니다")
                val srcDir = File(src)
                require(srcDir.exists()) { "pretrainCheckpointDir 존재하지 않음: $src" }
                loadModelWeights(File(srcDir, "model_weights.bin"))
                optimizer.resetState()
                iterationNumber = 0
                bestLoss = baselineLoss
                println("Pretrain 가중치 로드 완료: ${srcDir.absolutePath}; optimizer state reset; iter=0")
            }
            else -> error("알 수 없는 initFrom 값: ${config.initFrom}")
        }

        setTrainingMode(true)

        val startTime = System.currentTimeMillis()
        var runningLoss = 0.0
        val evalInterval = config.evalInterval.coerceAtLeast(1)

        while (iterationNumber <= config.maxIters) {
            optimizer.updateLearningRate(getLearningRate(iterationNumber))

            if (iterationNumber % evalInterval == 0) {
                setTrainingMode(false)
                val (trainLoss, valLoss) = estimateLoss()
                setTrainingMode(true)

                val avg = (trainLoss + valLoss) / 2.0
                println(
                    "스텝 $iterationNumber: " +
                        "훈련 ${formatLoss(trainLoss)} | 검증 ${formatLoss(valLoss)} | 평균 $avg" +
                        " | grad-norm ${"%.3f".format(lastGradNorm)}"
                )
                val isBest = avg < bestLoss
                if (isBest) bestLoss = avg
                if (isBest || config.alwaysSaveCheckpoint) saveCheckpoint(isBest)

                if (config.earlyStopPatience > 0) {
                    if (isBest) {
                        earlyStopCounter = 0
                    } else {
                        earlyStopCounter++
                        if (earlyStopCounter >= config.earlyStopPatience) {
                            println(
                                "Early stop: best 갱신 ${config.earlyStopPatience}회 연속 없음 — " +
                                    "iter=${iterationNumber}에서 종료 (maxIters=${config.maxIters})"
                            )
                            break
                        }
                    }
                }
            }

            if (iterationNumber == 0 && config.evalOnly) break

            val stepLoss = trainStep()
            lastGradNorm = if (config.gradClip > 0.0f) clipGradients(config.gradClip) else computeGradNorm()
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

    private fun trainStep(): Double {
        val accum = config.gradientAccumulationSteps
        val batch = config.batchSize
        val totalSeqs = accum * batch
        val upstreamGrad = 1.0f / totalSeqs

        val allSeqs = buildList<Pair<IntArray, IntArray>> {
            for (microStep in 0 until accum) {
                val (inputs, targets) = trainLoader.getBatch()
                for (b in inputs.indices) add(inputs[b] to targets[b])
            }
        }

        return if (workers.isEmpty()) trainStepSequential(allSeqs, upstreamGrad)
        else trainStepParallel(allSeqs, upstreamGrad)
    }

    private fun trainStepSequential(
        allSeqs: List<Pair<IntArray, IntArray>>,
        upstreamGrad: Float,
    ): Double {
        var totalLoss = 0.0
        for ((seqIdx, seq) in allSeqs.withIndex()) {
            val (input, target) = seq
            val logits = model.forward(input)
            val ce = turboCrossEntropyForward(logits, target, config.labelSmoothing, zLossWeight)
            if (!ce.loss.isFinite()) {
                error("학습 loss가 비정상(NaN/Inf): iter=$iterationNumber, seq=$seqIdx, value=${ce.loss}")
            }
            totalLoss += ce.loss.toDouble()
            val gLogits = turboCrossEntropyBackward(
                logits, target, ce.softmax, upstreamGrad, config.labelSmoothing,
                zLossWeight, ce.lsePerRow,
            )
            model.backward(gLogits)
        }
        return totalLoss / allSeqs.size
    }

    private fun trainStepParallel(
        allSeqs: List<Pair<IntArray, IntArray>>,
        upstreamGrad: Float,
    ): Double {
        for (worker in workers) {
            turboSyncParamsData(worker.parameters(), model.parameters())
            worker.zeroGrad()
        }

        val chunks = turboDistributeRoundRobin(allSeqs, workers.size)
        val perWorkerLoss = DoubleArray(workers.size)

        // Phase 5.1: ForkJoinPool work-stealing (coroutine 대비 CPU bound 핵심 루프에서 더 가벼운 dispatch)
        turboForkJoinIndices(chunks.size) { wi ->
            val worker = workers[wi]
            val chunk = chunks[wi]
            var localLoss = 0.0
            for ((input, target) in chunk) {
                val logits = worker.forward(input)
                val ce = turboCrossEntropyForward(logits, target, config.labelSmoothing, zLossWeight)
                if (!ce.loss.isFinite()) {
                    error("학습 loss가 비정상(NaN/Inf): iter=$iterationNumber, worker=$wi, value=${ce.loss}")
                }
                localLoss += ce.loss.toDouble()
                val gLogits = turboCrossEntropyBackward(
                    logits, target, ce.softmax, upstreamGrad, config.labelSmoothing,
                    zLossWeight, ce.lsePerRow,
                )
                worker.backward(gLogits)
            }
            perWorkerLoss[wi] = localLoss
        }

        model.zeroGrad()
        for (worker in workers) {
            turboAccumulateGrads(model.parameters(), worker.parameters())
        }

        return perWorkerLoss.sum() / allSeqs.size
    }

    private fun estimateLoss(): Pair<Double, Double> {
        val trainBatches = List(config.evalIters) { trainLoader.getBatch() }
        val valBatches = List(config.evalIters) { valLoader.getBatch() }

        return if (workers.isEmpty()) {
            val trainAvg = trainBatches.map { evaluateBatch(it.first, it.second) }.average()
            val valAvg = valBatches.map { evaluateBatch(it.first, it.second) }.average()
            trainAvg to valAvg
        } else {
            for (worker in workers) turboSyncParamsData(worker.parameters(), model.parameters())
            evaluateBatchesParallel(trainBatches) to evaluateBatchesParallel(valBatches)
        }
    }

    private fun evaluateBatch(inputs: Array<IntArray>, targets: Array<IntArray>): Double {
        var sum = 0.0
        for (b in inputs.indices) {
            val logits = model.forward(inputs[b])
            sum += turboCrossEntropyForward(logits, targets[b], zLossWeight = zLossWeight).loss.toDouble()
        }
        return sum / inputs.size
    }

    private fun evaluateBatchesParallel(
        batches: List<Pair<Array<IntArray>, Array<IntArray>>>,
    ): Double {
        val allSeqs = buildList<Pair<IntArray, IntArray>> {
            for ((inputs, targets) in batches) {
                for (b in inputs.indices) add(inputs[b] to targets[b])
            }
        }
        val chunks = turboDistributeRoundRobin(allSeqs, workers.size)
        val perWorkerLoss = DoubleArray(workers.size)

        turboForkJoinIndices(chunks.size) { wi ->
            val worker = workers[wi]
            val chunk = chunks[wi]
            var localLoss = 0.0
            for ((input, target) in chunk) {
                val logits = worker.forward(input)
                localLoss += turboCrossEntropyForward(logits, target, zLossWeight = zLossWeight).loss.toDouble()
            }
            perWorkerLoss[wi] = localLoss
        }
        return perWorkerLoss.sum() / allSeqs.size
    }

    private fun setTrainingMode(mode: Boolean) {
        model.setTraining(mode)
        for (w in workers) w.setTraining(mode)
    }

    private fun computeGradNorm(): Float {
        var sumSq = 0.0f
        for (p in model.parameters()) {
            val g = p.grad ?: continue
            for (i in g.indices) sumSq += g[i] * g[i]
        }
        return sqrt(sumSq)
    }

    private fun clipGradients(maxNorm: Float): Float {
        val totalNorm = computeGradNorm()
        if (totalNorm > maxNorm) {
            val scale = maxNorm / totalNorm
            for (p in model.parameters()) {
                val g = p.grad ?: continue
                for (i in g.indices) g[i] *= scale
            }
        }
        return totalNorm
    }

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

    private fun saveCheckpoint(isBest: Boolean) {
        val nextVersion = nextCheckpointVersion()
        val dir = File(modelPath, "v%04d".format(nextVersion))
        dir.mkdirs()

        val meta = TurboCheckpoint(
            iterationNumber = iterationNumber,
            bestValidationLoss = bestLoss,
            modelArgs = turboModelConfig,
            config = config,
        )
        val json = Json { prettyPrint = true; encodeDefaults = true }
        File(dir, "checkpoint.json").writeText(json.encodeToString(meta))
        saveModelWeights(File(dir, "model_weights.bin"))
        optimizer.saveState(File(dir, "optimizer_state.bin"))

        val srcMeta = File("${config.dataPath}/meta.json")
        if (srcMeta.exists()) srcMeta.copyTo(File(dir, "meta.json"), overwrite = true)

        val label = if (isBest) "best" else "always"
        println("체크포인트 저장 완료 ($label): ${File(dir, "checkpoint.json").absolutePath}")

        runSamplesForCheckpoint(dir)
    }

    private fun runSamplesForCheckpoint(ckptDir: File) {
        val prompts = listOf(
            "<|bos|>\\n# Apple\\n",
            "<|bos|>\\n# Cat\\n",
            "<|bos|>\\n# Run\\n",
            "<|bos|>\\n# Big\\n",
            "<|bos|>\\n# Tree\\n",
        )
        try {
            val sampleCfg = SampleConfig(
                modelDirectoryPath = ckptDir.absolutePath,
                numberOfSamples = 1,
                maximumNewTokens = 80,
                samplingTemperature = 0.0f,
                topKFilteringSize = 40,
                topProbabilityThreshold = 0.95f,
                repetitionPenalty = 1.15f,
                stopTokenIds = listOf(0),
            )
            val sampler = TurboSampler(sampleCfg)
            println("--- 샘플 (5 prompt × 1 sample, greedy T=0) ---")
            for (prompt in prompts) {
                val samples = sampler.generate(prompt)
                samples.forEach { s -> println("[$prompt] ${s.trim()}") }
            }
            println("--- 샘플 끝 ---")
        } catch (e: Exception) {
            println("샘플링 실패 (학습 계속): ${e.message}")
        }
    }

    private fun nextCheckpointVersion(): Int {
        val root = File(modelPath)
        if (!root.exists()) return 1
        val versionRegex = Regex("^v(\\d+)$")
        val maxVersion = (root.listFiles { f -> f.isDirectory } ?: emptyArray())
            .mapNotNull { versionRegex.matchEntire(it.name)?.groupValues?.get(1)?.toIntOrNull() }
            .maxOrNull() ?: 0
        return maxVersion + 1
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

    private fun findLatestCheckpoint(): File? {
        val root = File(modelPath)
        if (!root.exists()) return null
        val parser = Json { ignoreUnknownKeys = true }
        var bestDir: File? = null
        var bestIter = -1
        for (d in root.listFiles { f -> f.isDirectory } ?: emptyArray()) {
            val jsonFile = File(d, "checkpoint.json")
            val weightFile = File(d, "model_weights.bin")
            if (!jsonFile.exists() || !weightFile.exists()) continue
            val meta = try {
                parser.decodeFromString<TurboCheckpoint>(jsonFile.readText())
            } catch (_: Exception) {
                continue
            }
            if (meta.iterationNumber > bestIter) {
                bestIter = meta.iterationNumber
                bestDir = d
            }
        }
        return bestDir
    }

    private fun loadCheckpoint(dir: File) {
        val parser = Json { ignoreUnknownKeys = true }
        val meta = parser.decodeFromString<TurboCheckpoint>(File(dir, "checkpoint.json").readText())
        iterationNumber = meta.iterationNumber
        bestLoss = meta.bestValidationLoss
        loadModelWeights(File(dir, "model_weights.bin"))

        val optFile = File(dir, "optimizer_state.bin")
        if (optFile.exists()) {
            optimizer.loadState(optFile)
        } else {
            println("경고: optimizer_state.bin 없음 — 옵티마이저 모멘트는 0에서 재시작합니다.")
        }

        println(
            "체크포인트 재개: iter=$iterationNumber, bestLoss=${"%.4f".format(bestLoss)} " +
                "from ${dir.absolutePath}"
        )
    }

    private fun loadModelWeights(file: File) {
        require(file.exists()) { "가중치 파일 없음: ${file.absolutePath}" }
        file.inputStream().use { input ->
            val buf = ByteArray(4)
            for (p in model.parameters()) {
                for (i in 0 until p.numel) {
                    require(input.read(buf) == 4) { "가중치 파일 EOF 조기 도달" }
                    p.data[i] = ByteBuffer.wrap(buf).float
                }
            }
        }
    }

    private fun readVocabSize(): Int {
        val meta = File("${config.dataPath}/meta.json").readText()
        val parser = Json { ignoreUnknownKeys = true }
        return parser.decodeFromString<MetaInfo>(meta).vocabularySize
    }

    /** record-aware sampling용 — meta.json의 stringToIndex에서 BOS 토큰 id 추출. */
    private fun readBosTokenId(): Int {
        val meta = File("${config.dataPath}/meta.json").readText()
        val parser = Json { ignoreUnknownKeys = true }
        val info = parser.decodeFromString<MetaInfo>(meta)
        return info.stringToIndex[CharBPE.BOS_TOKEN]
            ?: error("meta.json에 ${CharBPE.BOS_TOKEN} 가 없음 (recordAwareSampling 사용 불가)")
    }

    private fun buildModelConfig(): TurboModelConfig {
        val gptCfg = GPTConfig(
            maxSequenceLength = config.blockSize,
            vocabularySize = vocabularySize,
            numberOfLayers = config.numberOfLayers,
            numberOfAttentionHeads = config.numberOfHeads,
            embeddingDimension = config.embeddingDimension,
            useBias = config.bias,
            dropoutProbability = config.dropout,
        )
        return TurboModelConfig(
            gpt = gptCfg,
            normalizationType = normalizationType,
            numKvHeads = numKvHeads,
            useQkNorm = useQkNorm,
            useFusedQkv = useFusedQkv,
            zLossWeight = zLossWeight,
            tieWeights = config.tieWeights,
            mlpActivation = config.mlpActivation,
            positionEncoding = config.positionEncoding,
        )
    }

    private fun formatLoss(loss: Double): String {
        val pct = maxOf(0.0, (baselineLoss - loss) / baselineLoss * 100)
        return "%.2f (%.1f%%)".format(loss, pct)
    }
}
