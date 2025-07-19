package train

import Value
import data.MetaInfo
import gpt.GPTConfig
import gpt.PikoGPT
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.coroutines.*
import sumOf
import java.io.File
import java.nio.ByteBuffer
import kotlin.io.path.Path
import kotlin.math.*
import kotlin.random.Random

// 메인 훈련 클래스
class Trainer(private val config: TrainConfig) {
    private lateinit var model: PikoGPT
    private lateinit var optimizer: AdamW
    private lateinit var trainLoader: DataLoader
    private lateinit var valLoader: DataLoader

    private var iterNum = 0
    private var bestValLoss = Double.MAX_VALUE
    private var baselineLoss = ln(getVocabSize().toDouble())
    val ds = "${config.calculateTotalParameters(getVocabSize())}"
    val path = "${config.modelDir}/${config.subDir ?: ds}"

    fun train() {
        println("=== PikoGPT 훈련 시작 ===")
        println("설정: $config")
        println("베이스라인 손실: ${baselineLoss.format(4)} (0% 진행률 기준)")

        // 디렉토리 생성
        Path(path).toFile().mkdirs()

        // 모델 초기화
        initModel()

        // 데이터 로더 초기화
        trainLoader = DataLoader("${config.dataPath}/train.bin", config.batchSize, config.blockSize)
        valLoader = DataLoader("${config.dataPath}/val.bin", config.batchSize, config.blockSize)

        // 옵티마이저 초기화
        optimizer = AdamW(
            params = model.parameters(),
            lr = config.learningRate,
            beta1 = config.beta1,
            beta2 = config.beta2,
            weightDecay = config.weightDecay
        )

        // 훈련 루프
        val startTime = System.currentTimeMillis()
        var runningLoss = 0.0

        while (iterNum <= config.maxIters) {
            // 학습률 조정
            val lr = getLearningRate(iterNum)
            optimizer.updateLearningRate(lr)

            // 평가
            if (iterNum % config.evalInterval == 0) {
                val losses = estimateLoss()
                val trainProgress = formatLossWithProgress(losses.first)
                val valProgress = formatLossWithProgress(losses.second)
                println("스텝 $iterNum: 훈련 $trainProgress | 검증 $valProgress")

                if (losses.second < bestValLoss || config.alwaysSaveCheckpoint) {
                    bestValLoss = losses.second
                    if (iterNum > 0) {
                        saveCheckpoint()
                    }
                }
            }

            if (iterNum == 0 && config.evalOnly) {
                break
            }

            // 미니배치 훈련
            var accLoss = 0.0
            for (microStep in 0 until config.gradientAccumulationSteps) {
                val (x, y) = trainLoader.getBatch()
                val loss = trainStep(x, y)
                accLoss += loss / config.gradientAccumulationSteps
            }

            // 그래디언트 클리핑
            if (config.gradClip > 0.0) {
                clipGradients(config.gradClip)
            }

            // 옵티마이저 스텝
            optimizer.step()
            optimizer.zeroGrad()

            // 로깅
            runningLoss = if (runningLoss == 0.0) accLoss else 0.9 * runningLoss + 0.1 * accLoss

            if (iterNum % config.logInterval == 0) {
                val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
                val lossProgress = formatLossWithProgress(runningLoss)
                println("반복 $iterNum: 손실 $lossProgress, elapsed ${elapsed.format(0)}s.")
            }

            iterNum++
        }

        println("\n훈련 완료!")
    }

    private fun initModel() {
        val modelConfig = GPTConfig(
            blockSize = config.blockSize,
            vocabSize = getVocabSize(),
            nLayer = config.nLayer,
            nHead = config.nHead,
            nEmbd = config.nEmbd,
            bias = config.bias,
            dropout = config.dropout
        )

        when (config.initFrom) {
            "scratch" -> {
                println("새 모델 초기화")
                model = PikoGPT(modelConfig)
            }

            "resume" -> {
                println("체크포인트에서 재개")
                loadCheckpoint()
            }

            else -> throw IllegalArgumentException("Unknown init_from: ${config.initFrom}")
        }

        println("모델 파라미터 수: ${model.parameters().size}")
    }

    private fun getVocabSize(): Int {
        // meta.json에서 vocab_size 읽기
        val metaFile = File("${config.dataPath}/meta.json")
        val json = Json { ignoreUnknownKeys = true }
        val meta = json.decodeFromString<MetaInfo>(metaFile.readText())
        return meta.vocabSize
    }

    private fun trainStep(x: Array<IntArray>, y: Array<IntArray>): Float {
        // 훈련 모드 설정
        gpt.Dropout.training = true
        
        // 순전파
        var totalLoss = Value(0.0f)

        for (b in x.indices) {
            val logits = model.forward(x[b])

            // Cross-entropy loss 계산
            for (t in y[b].indices) {
                val target = y[b][t]
                val logitVec = logits[t]

                // Softmax와 cross-entropy
                val maxLogit = logitVec.maxByOrNull { it.data } ?: Value(0.0f)
                val expLogits = logitVec.map { (it - maxLogit).exp() }
                val sumExp = expLogits.reduce { acc, v -> acc + v }

                val loss = (logitVec[target] - maxLogit).negate() + sumExp.log()
                totalLoss = totalLoss + loss
            }
        }

        val avgLoss = totalLoss * Value(1.0f / (x.size * y[0].size))

        // 역전파
        model.parameters().forEach { it.grad = 0.0f }
        avgLoss.backward()

        return avgLoss.data
    }

    private fun estimateLoss(): Pair<Double, Double> = runBlocking {
        val trainLossesDeferred = (0 until config.evalIters).map { 
            async(Dispatchers.Default) {
                val (x, y) = trainLoader.getBatch()
                evaluateBatch(x, y)
            }
        }
        
        val valLossesDeferred = (0 until config.evalIters).map { 
            async(Dispatchers.Default) {
                val (x, y) = valLoader.getBatch()
                evaluateBatch(x, y)
            }
        }

        val trainLosses = trainLossesDeferred.awaitAll()
        val valLosses = valLossesDeferred.awaitAll()

        return@runBlocking Pair(trainLosses.average(), valLosses.average())
    }

    private fun evaluateBatch(x: Array<IntArray>, y: Array<IntArray>): Double {
        // 평가 모드 설정 (dropout 비활성화)
        gpt.Dropout.training = false
        
        var totalLoss = 0.0

        for (b in x.indices) {
            val logits = model.forward(x[b])

            for (t in y[b].indices) {
                val target = y[b][t]
                val logitVec = logits[t]

                val maxLogit = logitVec.maxByOrNull { it.data }?.data ?: 0.0f
                val expSum = logitVec.sumOf { exp((it.data - maxLogit)) }

                totalLoss += -logitVec[target].data + maxLogit + ln(expSum)
            }
        }

        return totalLoss / (x.size * y[0].size)
    }

    private fun getLearningRate(iter: Int): Float {
        if (!config.decayLr) return config.learningRate

        // Warmup
        if (iter < config.warmupIters) {
            return config.learningRate * (iter + 1).toFloat() / config.warmupIters.toFloat()
        }

        // Min LR
        if (iter > config.lrDecayIters) {
            return config.minLr
        }

        // Cosine decay
        val decayRatio = (iter - config.warmupIters).toDouble() / (config.lrDecayIters - config.warmupIters)
        val coeff: Float = 0.5f * (1.0f + cos(PI * decayRatio).toFloat())
        return config.minLr + coeff * (config.learningRate - config.minLr)
    }

    private fun clipGradients(maxNorm: Float) {
        val params = model.parameters()
        
        // 직렬로 노름 계산 (작은 모델에서는 더 빠름)
        val totalNorm = sqrt(params.sumOf { param -> param.grad * param.grad })

        if (totalNorm > maxNorm) {
            val scale = maxNorm / totalNorm
            params.forEach { param ->
                param.grad *= scale
            }
        }
    }

    private fun saveCheckpoint() {
        println("체크포인트 저장 중...")

        // 모델 상태 추출
        val modelState = extractModelState()

        // 옵티마이저 상태 추출
        val optimizerState = OptimizerState(
            iteration = iterNum,
            m = mapOf(), // 실제로는 옵티마이저의 모멘트 값들을 저장해야 함
            v = mapOf()
        )

        // 체크포인트 생성
        val checkpoint = Checkpoint(
            modelState = modelState,
            optimizerState = optimizerState,
            modelArgs = model.config,
            iterNum = iterNum,
            bestValLoss = bestValLoss,
            config = config
        )

        // JSON으로 저장
        val json = Json {
            prettyPrint = true
            encodeDefaults = true
        }
        val lossInt = (bestValLoss * 10).toInt()
        File("${path}/${lossInt}").mkdir()
        val checkpointFile = File("${path}/${lossInt}/checkpoint.json")
        checkpointFile.writeText(json.encodeToString(checkpoint))

        // 가중치를 별도의 바이너리 파일로 저장 (더 효율적)
        saveModelWeights("${path}/${lossInt}/model_weights.bin")

        // meta.json 파일 복사
        val sourceMetaFile = File("${config.dataPath}/meta.json")
        val targetMetaFile = File("${path}/${lossInt}/meta.json")
        if (sourceMetaFile.exists()) {
            sourceMetaFile.copyTo(targetMetaFile, overwrite = true)
            println("meta.json 복사 완료")
        } else {
            println("경고: meta.json 파일을 찾을 수 없습니다: ${sourceMetaFile.absolutePath}")
        }

        println("체크포인트 저장 완료: ${checkpointFile.absolutePath}")
    }

    private fun extractModelState(): ModelState {
        // 간단한 구현을 위해 현재 가중치 값들을 직접 추출
        // 실제로는 모델의 각 레이어에서 가중치를 추출하는 메서드가 필요

        // 더미 구현 - 실제로는 모델의 파라미터를 추출해야 함
        return ModelState(
            tokenEmbedding = List(model.config.vocabSize) {
                List(model.config.nEmbd) { Random.nextDouble(-0.1, 0.1) }
            },
            positionEmbedding = List(model.config.blockSize) {
                List(model.config.nEmbd) { Random.nextDouble(-0.1, 0.1) }
            },
            blocks = List(model.config.nLayer) {
                BlockState(
                    ln1 = LayerNormState(
                        weight = List(model.config.nEmbd) { 1.0 },
                        bias = if (model.config.bias) List(model.config.nEmbd) { 0.0 } else null
                    ),
                    attn = AttentionState(
                        qProj = createLinearState(model.config.nEmbd, model.config.nEmbd),
                        kProj = createLinearState(model.config.nEmbd, model.config.nEmbd),
                        vProj = createLinearState(model.config.nEmbd, model.config.nEmbd),
                        outProj = createLinearState(model.config.nEmbd, model.config.nEmbd)
                    ),
                    ln2 = LayerNormState(
                        weight = List(model.config.nEmbd) { 1.0 },
                        bias = if (model.config.bias) List(model.config.nEmbd) { 0.0 } else null
                    ),
                    ffn = FeedForwardState(
                        cFc = createLinearState(model.config.nEmbd, 4 * model.config.nEmbd),
                        cProj = createLinearState(4 * model.config.nEmbd, model.config.nEmbd)
                    )
                )
            },
            lmHead = createLinearState(model.config.nEmbd, model.config.vocabSize)
        )
    }

    private fun createLinearState(inFeatures: Int, outFeatures: Int): LinearState {
        return LinearState(
            weight = List(outFeatures) {
                List(inFeatures) { Random.nextDouble(-0.1, 0.1) }
            },
            bias = if (config.bias) List(outFeatures) { 0.0 } else null
        )
    }

    private fun saveModelWeights(filename: String) {
        // 모델의 실제 Value 객체들의 data 값을 바이너리로 저장
        val file = File(filename)
        file.outputStream().use { stream ->
            model.parameters().forEach { param ->
                // Double을 바이트로 변환하여 저장
                val bytes = ByteBuffer.allocate(8).putFloat(param.data).array()
                stream.write(bytes)
            }
        }
    }

    private fun loadCheckpoint() {
        println("체크포인트 로드 중...")

        val checkpointFile = File("${path}/checkpoint.json")
        if (!checkpointFile.exists()) {
            throw IllegalStateException("체크포인트 파일을 찾을 수 없습니다: ${checkpointFile.absolutePath}")
        }

        val json = Json { ignoreUnknownKeys = true }
        val checkpoint = json.decodeFromString<Checkpoint>(checkpointFile.readText())

        // 모델 설정 확인
        val modelConfig = checkpoint.modelArgs
        model = PikoGPT(modelConfig)

        // 가중치 로드
        loadModelWeights("${path}/model_weights.bin")

        // 훈련 상태 복원
        iterNum = checkpoint.iterNum
        bestValLoss = checkpoint.bestValLoss

        println("체크포인트 로드 완료 (iteration: $iterNum, best val loss: $bestValLoss)")
    }

    private fun loadModelWeights(filename: String) {
        val file = File(filename)
        if (!file.exists()) {
            println("가중치 파일을 찾을 수 없습니다. 랜덤 초기화를 사용합니다.")
            return
        }

        file.inputStream().use { stream ->
            val params = model.parameters()
            val buffer = ByteArray(8)

            params.forEach { param ->
                if (stream.read(buffer) == 8) {
                    param.data = ByteBuffer.wrap(buffer).getFloat()
                }
            }
        }
    }

    // 확장 함수들
    fun Double.format(digits: Int): String = "%.${digits}f".format(this)
    fun Value.negate(): Value = this * Value(-1.0f)
    
    // 퍼센트 기반 손실 변환 함수
    private fun lossToPercentage(loss: Double): Double {
        return maxOf(0.0, (baselineLoss - loss) / baselineLoss * 100)
    }
    
    // 진행률 바 생성 함수
    private fun createProgressBar(percentage: Double, width: Int = 5): String {
        val filled = (percentage / 100.0 * width).toInt()
        val empty = width - filled
        return "▓".repeat(filled) + "░".repeat(empty)
    }
    
    // 손실과 퍼센트를 함께 표시하는 함수
    private fun formatLossWithProgress(loss: Double): String {
        val percentage = lossToPercentage(loss)
        val progressBar = createProgressBar(percentage)
        return "${loss.format(2)} (${percentage.format(1)}%) $progressBar"
    }

    fun Value.log(): Value {
        val output = Value(ln(this.data), setOf(this))

        output._backward = {
            this.grad += output.grad / this.data
        }

        return output
    }
}

