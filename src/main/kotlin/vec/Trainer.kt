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
        // 기본 상한을 **4**로 둔 이유: 12코어 머신에서 측정 결과 worker 4와 8이 같은
        // 0.5s/iter로 수렴 (공유 L3 캐시 + Apple Silicon P/E core mix 특성). 그 이상 쓰면
        // CPU만 더 쓰고 속도는 같으니 4가 비용 효율적. 머신 특성이 다르면
        // 환경변수 VEC_MAX_WORKERS로 override 가능 (벤치/튜닝 용도).
        val totalSeqsPerIter = config.batchSize * config.gradientAccumulationSteps
        val cpuCount = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)
        val envCap = System.getenv("VEC_MAX_WORKERS")?.toIntOrNull()?.coerceAtLeast(1)
        val defaultCap = 4
        val desiredWorkers = minOf(envCap ?: defaultCap, cpuCount, totalSeqsPerIter)
        workers = if (desiredWorkers >= 2) {
            println(
                "데이터 병렬 학습 활성 — worker 수 $desiredWorkers " +
                    "(CPU $cpuCount, seq/iter $totalSeqsPerIter" +
                    (if (envCap != null) ", VEC_MAX_WORKERS=$envCap" else "") +
                    ")"
            )
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

        if (config.initFrom == "resume") {
            val latest = findLatestCheckpoint()
            if (latest != null) {
                loadCheckpoint(latest)
            } else {
                println("initFrom=resume 이지만 체크포인트를 찾지 못함 — scratch로 진행합니다 (modelPath=$modelPath)")
            }
        }

        // 학습 모드 ON — dropout 활성화. eval 시점에만 잠시 OFF로 전환.
        setTrainingMode(true)

        val startTime = System.currentTimeMillis()
        var runningLoss = 0.0
        val evalInterval = config.evalInterval.coerceAtLeast(1)

        while (iterationNumber <= config.maxIters) {
            optimizer.updateLearningRate(getLearningRate(iterationNumber))

            if (iterationNumber % evalInterval == 0) {
                // eval 중엔 dropout 끄기 (master + workers 모두).
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

    /**
     * train/val 각각 [evalIters]개 배치 × batchSize 시퀀스를 forward해 평균 cross-entropy loss 계산.
     *
     * - `workers` 비어 있으면 마스터 순차.
     * - 아니면 `trainStepParallel`과 같은 분산 방식: 워커들에 master param을 1회 동기화한 뒤
     *   coroutine으로 동시 forward. Eval 중엔 weight 불변이라 grad/sync 필요 없음.
     */
    private fun estimateLoss(): Pair<Double, Double> {
        val trainBatches = List(config.evalIters) { trainLoader.getBatch() }
        val valBatches = List(config.evalIters) { valLoader.getBatch() }

        return if (workers.isEmpty()) {
            val trainAvg = trainBatches.map { evaluateBatch(it.first, it.second) }.average()
            val valAvg = valBatches.map { evaluateBatch(it.first, it.second) }.average()
            trainAvg to valAvg
        } else {
            // Eval 내내 worker param 불변이므로 train/val 공통 sync 1회.
            for (worker in workers) syncParamsData(worker.parameters(), model.parameters())
            evaluateBatchesParallel(trainBatches) to evaluateBatchesParallel(valBatches)
        }
    }

    private fun evaluateBatch(inputs: Array<IntArray>, targets: Array<IntArray>): Double {
        var sum = 0.0
        for (b in inputs.indices) {
            val logits = model.forward(inputs[b])
            sum += crossEntropyForward(logits, targets[b]).loss.toDouble()
        }
        return sum / inputs.size
    }

    /** 배치 묶음을 모두 시퀀스로 flatten 후 worker들에 round-robin 분배해 coroutine forward. */
    private fun evaluateBatchesParallel(
        batches: List<Pair<Array<IntArray>, Array<IntArray>>>,
    ): Double {
        val allSeqs = buildList<Pair<IntArray, IntArray>> {
            for ((inputs, targets) in batches) {
                for (b in inputs.indices) add(inputs[b] to targets[b])
            }
        }
        val chunks = distributeRoundRobin(allSeqs, workers.size)
        val perWorkerLoss = DoubleArray(workers.size)

        runBlocking {
            coroutineScope {
                chunks.forEachIndexed { wi, chunk ->
                    launch(Dispatchers.Default) {
                        val worker = workers[wi]
                        var localLoss = 0.0
                        for ((input, target) in chunk) {
                            val logits = worker.forward(input)
                            localLoss += crossEntropyForward(logits, target).loss.toDouble()
                        }
                        perWorkerLoss[wi] = localLoss
                    }
                }
            }
        }
        return perWorkerLoss.sum() / allSeqs.size
    }

    /** master + workers의 모든 Dropout 레이어 training 플래그를 일괄 설정. */
    private fun setTrainingMode(mode: Boolean) {
        model.setTraining(mode)
        for (w in workers) w.setTraining(mode)
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
        optimizer.saveState(File(dir, "optimizer_state.bin"))

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

    /**
     * `modelPath` 하위에서 `checkpoint.json` + `model_weights.bin`을 모두 가진 디렉토리 중
     * 저장된 `iterationNumber`가 가장 큰 것을 반환. (항상 저장(`alwaysSaveCheckpoint`) 모드에서는
     * 디렉토리 이름이 best loss가 아니라 **당시 avg**로 결정되므로 여러 디렉토리가 누적되는데,
     * 그 중 가장 마지막 이터레이션을 고른다.)
     */
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
                parser.decodeFromString<VCheckpointMeta>(jsonFile.readText())
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

    /**
     * 체크포인트 디렉토리에서 iter/bestLoss, 모델 가중치, 옵티마이저 상태(있으면)를 읽어
     * 현재 학습 상태로 주입. 옵티마이저 상태 파일이 없는 구버전 체크포인트면 moment/timeStep만
     * 초기값 그대로 유지(=재개 이후 몇 iter는 warm-up 효과가 약해질 수 있으나 치명적이진 않음).
     */
    private fun loadCheckpoint(dir: File) {
        val parser = Json { ignoreUnknownKeys = true }
        val meta = parser.decodeFromString<VCheckpointMeta>(File(dir, "checkpoint.json").readText())
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
        return parser.decodeFromString<MetaInfo>(meta).vocabSize
    }

    private fun buildModelConfig(): GPTConfig = GPTConfig(
        maxSequenceLength = config.blockSize,
        vocabularySize = vocabularySize,
        numberOfLayers = config.numberOfLayers,
        numberOfAttentionHeads = config.numberOfHeads,
        embeddingDimension = config.embeddingDimension,
        useBias = config.bias,
        dropoutProbability = config.dropout,
    )

    private fun formatLoss(loss: Double): String {
        val pct = maxOf(0.0, (baselineLoss - loss) / baselineLoss * 100)
        return "%.2f (%.1f%%)".format(loss, pct)
    }
}
