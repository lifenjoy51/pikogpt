package vec

import data.MetaInfo
import gpt.GPTConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
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

    /**
     * 데이터 병렬 학습용 worker 복제본들. 매 iter 시작에 master로부터 param.data를 sync받고,
     * 시퀀스를 나눠 forward/backward 한 뒤 grad를 master로 merge한다.
     *
     * worker 수 = min(CPU 수, 한 iter의 총 시퀀스 수). 시퀀스가 1개면 오버헤드만 있으므로
     * 빈 리스트로 두고 순차 경로를 쓴다.
     */
    private lateinit var workers: List<PikoGPT>

    private val datasetSize: String by lazy {
        config.calculateTotalParameters(vocabularySize).toString()
    }
    private val modelPath: String by lazy { "${config.modelDir}/vec/$datasetSize" }
    private val baselineLoss: Double by lazy { ln(vocabularySize.toDouble()) }
    private var bestLoss: Double = 0.0
    private var iterationNumber: Int = 0
    private var vocabularySize: Int = 0

    /** 직전 optimizer step의 pre-clip gradient L2 norm — eval 로그에 함께 표시해 학습 안정성 진단. */
    private var lastGradNorm: Float = 0.0f

    fun train() {
        println("=== PikoGPT (vec 백엔드) 훈련 시작 ===")
        println("설정: $config")
        vocabularySize = readVocabSize()
        bestLoss = baselineLoss
        println("베이스라인 손실: ${"%.4f".format(baselineLoss)} (vocab=$vocabularySize)")

        Path(modelPath).toFile().mkdirs()

        model = PikoGPT(buildModelConfig())
        println("모델 파라미터 텐서 수: ${model.parameters().size}, 총 스칼라 원소: ${model.parameters().sumOf { it.numel }}")

        // Worker 복제본 준비 (data-parallel). 한 iter당 seq 개수보다 많을 이유 없음.
        val totalSeqsPerIter = config.batchSize * config.gradientAccumulationSteps
        val cpuCount = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)
        val desiredWorkers = minOf(cpuCount, totalSeqsPerIter)
        workers = if (desiredWorkers >= 2) {
            println("데이터 병렬 학습 활성 — worker 수 $desiredWorkers (CPU $cpuCount, seq/iter $totalSeqsPerIter)")
            List(desiredWorkers) { PikoGPT(buildModelConfig()) }
        } else {
            println("데이터 병렬 비활성 (worker=$desiredWorkers). 순차 경로 사용.")
            emptyList()
        }

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
                        "훈련 ${formatLoss(trainLoss)} | 검증 ${formatLoss(valLoss)} | 평균 $avg" +
                        " | grad-norm ${"%.3f".format(lastGradNorm)}"
                )
                val isBest = avg < bestLoss
                if (isBest) bestLoss = avg
                if (isBest || config.alwaysSaveCheckpoint) saveCheckpoint(avg, isBest)
            }

            if (iterationNumber == 0 && config.evalOnly) break

            // 한 optimizer step: accum 만큼 gradient를 누적
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

    /**
     * Gradient accumulation: [config.gradientAccumulationSteps] × [config.batchSize] 개의 시퀀스에 대해
     * cross-entropy 평균 loss의 기울기를 누적한다. upstream 스케일은 `1 / (accum * batch)`.
     *
     * [workers]가 비어 있지 않으면 **데이터 병렬 실행**: 시퀀스를 worker들에 round-robin으로
     * 나누고 Kotlin coroutines + Dispatchers.Default로 동시에 forward/backward, grad는 worker별
     * 로컬에 누적됐다가 마지막에 master로 merge. 시퀀스 1개뿐이거나 CPU 1개면 순차 경로.
     */
    private fun trainStep(): Double {
        val accum = config.gradientAccumulationSteps
        val batch = config.batchSize
        val totalSeqs = accum * batch
        val upstreamGrad = 1.0f / totalSeqs

        // 이 iter에 쓸 시퀀스 전체를 미리 뽑아둠 (DataLoader는 싱글 스레드에서만 호출).
        val allSeqs = buildList<Pair<IntArray, IntArray>> {
            for (microStep in 0 until accum) {
                val (inputs, targets) = trainLoader.getBatch()
                for (b in inputs.indices) add(inputs[b] to targets[b])
            }
        }

        return if (workers.isEmpty()) trainStepSequential(allSeqs, upstreamGrad)
        else trainStepParallel(allSeqs, upstreamGrad)
    }

    /** 순차 버전 — workers 비어 있을 때 사용. */
    private fun trainStepSequential(
        allSeqs: List<Pair<IntArray, IntArray>>,
        upstreamGrad: Float,
    ): Double {
        var totalLoss = 0.0
        for ((seqIdx, seq) in allSeqs.withIndex()) {
            val (input, target) = seq
            val logits = model.forward(input)
            val ce = crossEntropyForward(logits, target)
            if (!ce.loss.isFinite()) {
                error("학습 loss가 비정상(NaN/Inf): iter=$iterationNumber, seq=$seqIdx, value=${ce.loss}")
            }
            totalLoss += ce.loss.toDouble()
            val gLogits = crossEntropyBackward(logits, target, ce.softmax, upstreamGrad)
            model.backward(gLogits)
        }
        return totalLoss / allSeqs.size
    }

    /** 병렬 버전 — worker 복제본들에 시퀀스 분배 후 coroutine으로 동시 실행. */
    private fun trainStepParallel(
        allSeqs: List<Pair<IntArray, IntArray>>,
        upstreamGrad: Float,
    ): Double {
        // 1) worker params를 master와 동기화 + worker grads 초기화
        for (worker in workers) {
            syncParamsData(worker.parameters(), model.parameters())
            worker.zeroGrad()
        }

        // 2) round-robin 분배
        val chunks = distributeRoundRobin(allSeqs, workers.size)
        val perWorkerLoss = DoubleArray(workers.size)

        // 3) 동시 실행
        runBlocking {
            coroutineScope {
                chunks.forEachIndexed { wi, chunk ->
                    launch(Dispatchers.Default) {
                        val worker = workers[wi]
                        var localLoss = 0.0
                        for ((input, target) in chunk) {
                            val logits = worker.forward(input)
                            val ce = crossEntropyForward(logits, target)
                            if (!ce.loss.isFinite()) {
                                error("학습 loss가 비정상(NaN/Inf): iter=$iterationNumber, worker=$wi, value=${ce.loss}")
                            }
                            localLoss += ce.loss.toDouble()
                            val gLogits = crossEntropyBackward(logits, target, ce.softmax, upstreamGrad)
                            worker.backward(gLogits)
                        }
                        perWorkerLoss[wi] = localLoss
                    }
                }
            }
        }

        // 4) worker grads → master grad 합산 (master.grad는 옵티마이저가 읽음)
        model.zeroGrad()
        for (worker in workers) {
            accumulateGrads(model.parameters(), worker.parameters())
        }

        return perWorkerLoss.sum() / allSeqs.size
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

    /** 모든 파라미터 grad의 L2 norm을 계산해서 반환. */
    private fun computeGradNorm(): Float {
        var sumSq = 0.0f
        for (p in model.parameters()) {
            val g = p.grad ?: continue
            for (i in g.indices) sumSq += g[i] * g[i]
        }
        return sqrt(sumSq)
    }

    /** L2 norm 기반 gradient clipping. 반환값은 **clip 전** total norm — 진단용. */
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

    /**
     * 체크포인트를 `${modelPath}/{(loss*10).toInt()}/`에 저장.
     *
     * @param loss   저장 경로 결정용 loss (best 갱신 시 새 best, alwaysSaveCheckpoint 시 현재 avg)
     * @param isBest best validation loss 갱신 여부 — 로그에만 사용
     */
    private fun saveCheckpoint(loss: Double, isBest: Boolean) {
        val lossInteger = (loss * 10).toInt()
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

        val label = if (isBest) "best" else "always"
        println("체크포인트 저장 완료 ($label): ${File(dir, "checkpoint.json").absolutePath}")
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
