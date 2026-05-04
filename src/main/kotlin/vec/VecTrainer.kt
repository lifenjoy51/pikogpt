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
import sample.SampleConfig
import train.BatchSource
import train.DataLoader
import train.MixedDataLoader
import train.TrainConfig
import train.TripleDataLoader
import vec.layer.VecPikoGPT
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
 * 체크포인트 경로 규약: `${config.modelDir}/${datasetName}/vec/${paramCount}/v0001/` (4자리 zero-pad 버전).
 * 학습 시작/재개 시 기존 `v\d+/` 디렉터리를 스캔해 가장 큰 번호 +1로 새 디렉터리 생성.
 * 레거시 형식(`28/`, `34/` 같은 `bestLoss*10` 정수)도 loader가 인식 — 마이그레이션 강제 안 함.
 * (datasetName은 `config.dataPath`의 마지막 segment. 예: `data/tinyhelen-textbook` → `tinyhelen-textbook`.
 * 다른 데이터셋을 같은 `modelDir` 아래에서 자동으로 격리.)
 * 파일 3개:
 *   - `checkpoint.json`  : [VecCheckpoint] 직렬화
 *   - `model_weights.bin`: 모든 param의 float32를 big-endian 연속 덤프
 *   - `meta.json`        : 학습 데이터 dir의 vocab 메타 복사
 */
class VecTrainer(private val config: TrainConfig) {

    private lateinit var model: VecPikoGPT
    private lateinit var optimizer: VecAdamW
    private lateinit var trainLoader: BatchSource
    private lateinit var valLoader: DataLoader

    /**
     * 데이터 병렬 학습용 worker 복제본들. 매 iter 시작에 master로부터 param.data를 sync받고,
     * 시퀀스를 나눠 forward/backward 한 뒤 grad를 master로 merge한다.
     *
     * worker 수 = min(CPU 수, 한 iter의 총 시퀀스 수). 시퀀스가 1개면 오버헤드만 있으므로
     * 빈 리스트로 두고 순차 경로를 쓴다.
     */
    private lateinit var workers: List<VecPikoGPT>

    /** `config.dataPath`의 마지막 segment. 데이터셋별로 체크포인트 트리를 분리하는 키. */
    private val datasetName: String by lazy {
        config.dataPath.trimEnd('/').substringAfterLast('/')
    }

    /**
     * 모델 실제 파라미터 합 — `model.parameters().sumOf { numel }`. tied weight 등으로
     * `TrainConfig.calculateTotalParameters` 기댓값과 다를 수 있어 ckpt 경로 격리에는
     * 실측을 쓴다. `train()`에서 모델 생성 후 채워짐.
     */
    private lateinit var modelPath: String
    private val baselineLoss: Double by lazy { ln(vocabularySize.toDouble()) }
    private var bestLoss: Double = 0.0
    private var iterationNumber: Int = 0
    private var vocabularySize: Int = 0

    /** 직전 optimizer step의 pre-clip gradient L2 norm — eval 로그에 함께 표시해 학습 안정성 진단. */
    private var lastGradNorm: Float = 0.0f

    /** Early stop용 — best 갱신 안 된 eval 횟수. config.earlyStopPatience 도달 시 학습 종료. */
    private var earlyStopCounter: Int = 0

    fun train() {
        println("=== VecPikoGPT (vec 백엔드) 훈련 시작 ===")
        println("설정: $config")
        vocabularySize = readVocabSize()
        bestLoss = baselineLoss
        println("베이스라인 손실: ${"%.4f".format(baselineLoss)} (vocab=$vocabularySize)")

        model = VecPikoGPT(buildModelConfig())
        val actualParamCount = model.parameters().sumOf { it.numel }
        println("모델 파라미터 텐서 수: ${model.parameters().size}, 총 스칼라 원소: $actualParamCount")
        modelPath = "${config.modelDir}/$datasetName/vec/$actualParamCount"
        Path(modelPath).toFile().mkdirs()

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
            List(desiredWorkers) { VecPikoGPT(buildModelConfig()) }
        } else {
            println("데이터 병렬 비활성 (worker=$desiredWorkers). 순차 경로 사용.")
            emptyList()
        }

        // replay 경로가 지정되면 두 코퍼스(MixedDataLoader) 또는 세 코퍼스(TripleDataLoader)로 mix.
        // - replayDataPath만: IT(finetune) 단계의 BASE replay (2-way)
        // - replayDataPath + replayDataPath2: 3-stage curriculum 마지막 단계 multi-replay (3-way)
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
            else -> DataLoader("${config.dataPath}/train.bin", config.batchSize, config.blockSize)
        }
        valLoader = DataLoader("${config.dataPath}/val.bin", config.batchSize, config.blockSize)

        optimizer = VecAdamW(
            parameters = model.parameters(),
            learningRate = config.learningRate,
            beta1 = config.beta1,
            beta2 = config.beta2,
            weightDecay = config.weightDecay,
        )

        when (config.initFrom) {
            "scratch" -> { /* 신규 학습 — 추가 작업 없음 */ }
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

                // Early stop — best 갱신 N번 연속 없으면 학습 조기 종료. plateau 진입 후 시간 낭비 방지.
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
            val ce = crossEntropyForward(logits, target, config.labelSmoothing)
            if (!ce.loss.isFinite()) {
                error("학습 loss가 비정상(NaN/Inf): iter=$iterationNumber, seq=$seqIdx, value=${ce.loss}")
            }
            totalLoss += ce.loss.toDouble()
            val gLogits = crossEntropyBackward(logits, target, ce.softmax, upstreamGrad, config.labelSmoothing)
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
                            val ce = crossEntropyForward(logits, target, config.labelSmoothing)
                            if (!ce.loss.isFinite()) {
                                error("학습 loss가 비정상(NaN/Inf): iter=$iterationNumber, worker=$wi, value=${ce.loss}")
                            }
                            localLoss += ce.loss.toDouble()
                            val gLogits = crossEntropyBackward(logits, target, ce.softmax, upstreamGrad, config.labelSmoothing)
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
     * 체크포인트를 `${modelPath}/v0001/`(또는 다음 버전)에 저장.
     *
     * 디렉터리 이름은 `v` + 4자리 zero-pad 버전 번호. 매 저장마다 +1.
     * 기존 레거시 디렉터리(`28/` 같은 `bestLoss*10`)는 보존하고 무시.
     *
     * @param loss   저장 시점 loss — `checkpoint.json`의 `bestValidationLoss`에 보존
     * @param isBest best validation loss 갱신 여부 — 로그에만 사용
     */
    private fun saveCheckpoint(loss: Double, isBest: Boolean) {
        val nextVersion = nextCheckpointVersion()
        val dir = File(modelPath, "v%04d".format(nextVersion))
        dir.mkdirs()

        val meta = VecCheckpoint(
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

        // ckpt 저장 직후 sample 출력 — 학습 진행 상황을 정성적으로 추적.
        // 새 VecSampler가 ckpt를 다시 로드해 dropout-off 상태 보장. 비용은 무시 가능 (1M params).
        runSamplesForCheckpoint(dir)
    }

    /**
     * `ckpt` 디렉터리에서 `VecSampler`를 새로 만들어 5 prompt × 1 sample = 5개 응답을 stdout에 출력.
     * 매 ckpt 저장 시 학습 stdout 로그에 함께 찍혀 정성적 진행 추적이 가능.
     *
     * **temperature=0 (greedy)** — 모델의 진짜 belief를 보기 위해 random sampling 비활성.
     * 같은 ckpt에서 항상 같은 결과 → 의미 매핑 변화 추적이 명확.
     *
     * 비용: 5 prompt × 1 sample × ~80 token ≈ 400 forward (~5초/ckpt). 학습 시간 미미.
     * 실패하면 학습은 계속 진행 (예외 swallow).
     */
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
                numberOfSamples = 1,           // greedy는 deterministic이라 1번이면 충분
                maximumNewTokens = 80,
                samplingTemperature = 0.0f,    // greedy — 의미 매핑 정확 추적
                topKFilteringSize = 40,
                topProbabilityThreshold = 0.95f,
                repetitionPenalty = 1.15f,
                stopTokenIds = listOf(0),      // EOS만 — turnId는 dict 도메인 OOD라 사용 안 함
            )
            val sampler = VecSampler(sampleCfg)
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

    /**
     * 다음 체크포인트 버전 번호 — `${modelPath}` 아래 기존 `v\d+` 디렉터리 중 최대 번호 + 1.
     * 디렉터리가 없거나 `v\d+` 패턴이 없으면 1.
     */
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
                parser.decodeFromString<VecCheckpoint>(jsonFile.readText())
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
        val meta = parser.decodeFromString<VecCheckpoint>(File(dir, "checkpoint.json").readText())
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

    private fun buildModelConfig(): GPTConfig = GPTConfig(
        maxSequenceLength = config.blockSize,
        vocabularySize = vocabularySize,
        numberOfLayers = config.numberOfLayers,
        numberOfAttentionHeads = config.numberOfHeads,
        embeddingDimension = config.embeddingDimension,
        useBias = config.bias,
        dropoutProbability = config.dropout,
        tieWeights = config.tieWeights,
        mlpActivation = config.mlpActivation,
        positionEncoding = config.positionEncoding,
    )

    private fun formatLoss(loss: Double): String {
        val pct = maxOf(0.0, (baselineLoss - loss) / baselineLoss * 100)
        return "%.2f (%.1f%%)".format(loss, pct)
    }
}
