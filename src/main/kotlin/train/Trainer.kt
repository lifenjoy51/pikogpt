package train

import Value
import data.MetaInfo
import gpt.Dropout
import gpt.GPTConfig
import gpt.PikoGPT
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import sumOf
import java.io.File
import java.nio.ByteBuffer
import kotlin.io.path.Path
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * 메인 훈련 클래스
 *
 * GPT 모델의 전체 훈련 파이프라인을 관리합니다.
 * 모델 초기화, 데이터 로딩, 훈련 루프, 평가, 체크포인트 저장 등을 처리합니다.
 *
 * @param config 훈련에 대한 모든 설정을 포함하는 TrainConfig 객체
 */
class Trainer(private val config: TrainConfig) {
    /** GPT 모델 인스턴스 */
    private lateinit var model: PikoGPT

    /** AdamW 옵티마이저 인스턴스 */
    private lateinit var optimizer: AdamW

    /** 훈련 데이터 로더 */
    private lateinit var trainingDataLoader: DataLoader

    /** 검증 데이터 로더 */
    private lateinit var validationDataLoader: DataLoader

    /** 데이터셋 크기 식별자 (모델 파라미터 수 기반) */
    private val datasetSize = "${config.calculateTotalParameters(getVocabularySize())}"

    /** 모델 저장 경로 */
    private val modelPath = "${config.modelDir}/${config.subDir ?: datasetSize}"

    /** 베이스라인 손실 - 랜덤 추측의 이론적 손실 값 (ln(어휘수)) */
    private val baselineLoss = ln(getVocabularySize().toDouble())

    /** 지금까지의 최고 검증 성능 (손실이 작을수록 좋음) */
    private var bestLoss = baselineLoss

    /** 현재 훈련 이터레이션 번호 */
    private var iterationNumber = 0

    /**
     * 메인 훈련 프로세스 실행
     *
     * 전체 훈련 파이프라인을 실행합니다:
     * 1. 모델 및 데이터 로더 초기화
     * 2. 옵티마이저 설정
     * 3. 훈련 루프 실행 (평가, 배치 훈련, 체크포인트 저장)
     */
    fun train() {
        println("=== PikoGPT 훈련 시작 ===")
        println("설정: $config")
        println("베이스라인 손실: ${baselineLoss.format(4)} (0% 진행률 기준)")

        // 디렉토리 생성
        Path(modelPath).toFile().mkdirs()

        // 모델 초기화
        initModel()

        // 데이터 로더 초기화
        trainingDataLoader = DataLoader("${config.dataPath}/train.bin", config.batchSize, config.blockSize)
        validationDataLoader = DataLoader("${config.dataPath}/val.bin", config.batchSize, config.blockSize)

        // 옵티마이저 초기화
        optimizer = AdamW(
            parameters = model.parameters(),
            learningRate = config.learningRate,
            beta1 = config.beta1,
            beta2 = config.beta2,
            weightDecay = config.weightDecay
        )

        // 훈련 루프
        val startTime = System.currentTimeMillis()
        var runningLoss = 0.0

        while (iterationNumber <= config.maxIters) {
            // 학습률 조정
            val currentLearningRate = getLearningRate(iterationNumber)
            optimizer.updateLearningRate(currentLearningRate)

            // 평가
            if (iterationNumber % config.evalInterval == 0) {
                val estimatedLosses = estimateLoss()
                val lossValue = (estimatedLosses.first + estimatedLosses.second) / 2
                val trainingProgress = formatLossWithProgress(estimatedLosses.first)
                val validationProgress = formatLossWithProgress(estimatedLosses.second)
                println("스텝 $iterationNumber: 훈련 $trainingProgress | 검증 $validationProgress | 평균 $lossValue")
                
                if (lossValue < bestLoss) {
                    bestLoss = lossValue
                    saveCheckpoint()
                }
            }

            if (iterationNumber == 0 && config.evalOnly) {
                break
            }

            // 미니배치 훈련
            var accumulatedLoss = 0.0
            for (microStep in 0 until config.gradientAccumulationSteps) {
                val (inputSequences, targetSequences) = trainingDataLoader.getBatch()
                val stepLoss = trainStep(inputSequences, targetSequences)
                accumulatedLoss += stepLoss / config.gradientAccumulationSteps
            }

            // 그래디언트 클리핑
            if (config.gradClip > 0.0) {
                clipGradients(config.gradClip)
            }

            // 옵티마이저 스텝
            optimizer.step()
            optimizer.zeroGrad()

            // 로깅
            runningLoss = if (runningLoss == 0.0) accumulatedLoss else 0.9 * runningLoss + 0.1 * accumulatedLoss

            if (iterationNumber % config.logInterval == 0) {
                val elapsedTime = (System.currentTimeMillis() - startTime) / 1000.0
                val lossProgress = formatLossWithProgress(runningLoss)
                println("반복 $iterationNumber: 손실 $lossProgress, elapsed ${elapsedTime.format(0)}s.")
            }

            iterationNumber++
        }

        println("\n훈련 완료!")
    }

    /**
     * 모델 초기화
     *
     * 설정에 따라 새 모델을 생성하거나 체크포인트에서 모델을 로드합니다.
     * GPT 설정(레이어 수, 헤드 수, 임베딩 차원 등)을 사용하여 모델을 구성합니다.
     */
    private fun initModel() {
        val modelConfig = GPTConfig(
            maxSequenceLength = config.blockSize,
            vocabularySize = getVocabularySize(),
            numberOfLayers = config.numberOfLayers,
            numberOfAttentionHeads = config.numberOfHeads,
            embeddingDimension = config.embeddingDimension,
            useBias = config.bias,
            dropoutProbability = config.dropout
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

    /**
     * 어휘 사전 크기 가져오기
     *
     * 데이터 디렉토리의 meta.json 파일에서 어휘 사전 크기를 읽어옵니다.
     * 이 값은 모델의 출력 레이어 크기를 결정하는데 사용됩니다.
     *
     * @return 어휘 사전의 크기 (예: shakespeare_char의 경우 65)
     */
    private fun getVocabularySize(): Int {
        // meta.json에서 vocab_size 읽기
        val metadataFile = File("${config.dataPath}/meta.json")
        val jsonParser = Json { ignoreUnknownKeys = true }
        val metadata = jsonParser.decodeFromString<MetaInfo>(metadataFile.readText())
        return metadata.vocabSize
    }

    /**
     * 단일 훈련 스텝 실행
     *
     * 한 번의 미니배치에 대해 순전파, 손실 계산, 역전파를 수행합니다.
     * 입력 시퀀스로부터 다음 토큰을 예측하는 Cross-Entropy 손실을 계산합니다.
     *
     * @param inputSequences 입력 토큰 시퀀스 배열 [batch_size, block_size]
     * @param targetSequences 타겟 토큰 시퀀스 배열 [batch_size, block_size]
     * @return 평균 손실 값
     */
    private fun trainStep(inputSequences: Array<IntArray>, targetSequences: Array<IntArray>): Float {
        // 훈련 모드 설정
        Dropout.training = true
        
        // 순전파
        var totalLoss = 0.0f

        for (batchIndex in inputSequences.indices) {
            val logits = model.forward(inputSequences[batchIndex])

            // Cross-entropy loss 계산
            for (tokenIndex in targetSequences[batchIndex].indices) {
                val targetToken = targetSequences[batchIndex][tokenIndex]
                val logitVector = logits[tokenIndex]

                // Softmax와 cross-entropy
                val maxLogit = logitVector.maxByOrNull { it.scalarValue } ?: Value(0.0f)
                val exponentialLogits = logitVector.map { (it - maxLogit).exp() }
                val sumExponential = exponentialLogits.reduce { accumulator, value -> accumulator + value }

                val stepLoss = (logitVector[targetToken] - maxLogit).negate() + sumExponential.log()
                totalLoss += stepLoss.scalarValue
            }
        }

        val averageLoss = Value(totalLoss / (inputSequences.size * targetSequences[0].size))

        // 역전파
        model.parameters().forEach { it.gradient = 0.0f }
        averageLoss.backward()

        return averageLoss.scalarValue
    }

    /**
     * 훈련/검증 손실 추정
     *
     * 여러 배치에 대해 손실을 계산하고 평균을 내어 현재 모델의 성능을 평가합니다.
     * 병렬 처리를 사용하여 평가 속도를 향상시킵니다.
     *
     * @return Pair<훈련_손실, 검증_손실>
     */
    private fun estimateLoss(): Pair<Double, Double> = runBlocking {
        val trainingLossesDeferred = (0 until config.evalIters).map {
            async(Dispatchers.Default) {
                val (inputSequences, targetSequences) = trainingDataLoader.getBatch()
                evaluateBatch(inputSequences, targetSequences)
            }
        }

        val validationLossesDeferred = (0 until config.evalIters).map {
            async(Dispatchers.Default) {
                val (inputSequences, targetSequences) = validationDataLoader.getBatch()
                evaluateBatch(inputSequences, targetSequences)
            }
        }

        val trainingLosses = trainingLossesDeferred.awaitAll()
        val validationLosses = validationLossesDeferred.awaitAll()

        return@runBlocking Pair(trainingLosses.average(), validationLosses.average())
    }

    /**
     * 단일 배치 평가
     *
     * 훈련과 달리 dropout을 비활성화하고, 그래디언트 계산 없이 순전파만 수행합니다.
     * 평가 모드에서의 실제 성능을 측정하기 위해 사용됩니다.
     *
     * @param inputSequences 입력 토큰 시퀀스 배열
     * @param targetSequences 타겟 토큰 시퀀스 배열
     * @return 배치의 평균 손실
     */
    private fun evaluateBatch(inputSequences: Array<IntArray>, targetSequences: Array<IntArray>): Double {
        // 평가 모드 설정 (dropout 비활성화)
        gpt.Dropout.training = false
        
        var totalLoss = 0.0

        for (batchIndex in inputSequences.indices) {
            val logits = model.forward(inputSequences[batchIndex])

            for (tokenIndex in targetSequences[batchIndex].indices) {
                val targetToken = targetSequences[batchIndex][tokenIndex]
                val logitVector = logits[tokenIndex]

                val maxLogit = logitVector.maxByOrNull { it.scalarValue }?.scalarValue ?: 0.0f
                val exponentialSum = logitVector.sumOf { exp((it.scalarValue - maxLogit)) }

                totalLoss += -logitVector[targetToken].scalarValue + maxLogit + ln(exponentialSum)
            }
        }

        return totalLoss / (inputSequences.size * targetSequences[0].size)
    }

    /**
     * 학습률 스케줄링
     *
     * 훈련 진행에 따라 학습률을 조정합니다:
     * 1. Warmup: 초기에 점진적으로 학습률 증가
     * 2. Cosine Decay: 코사인 함수로 학습률 감소
     * 3. Minimum LR: 최소값 유지
     *
     * @param iteration 현재 이터레이션 번호
     * @return 조정된 학습률
     */
    private fun getLearningRate(iteration: Int): Float {
        if (!config.decayLr) return config.learningRate

        // Warmup
        if (iteration < config.warmupIters) {
            return config.learningRate * (iteration + 1).toFloat() / config.warmupIters.toFloat()
        }

        // Min LR
        if (iteration > config.learningRateDecayIterations) {
            return config.minimumLearningRate
        }

        // Cosine decay
        val decayRatio = (iteration - config.warmupIters).toDouble() / (config.learningRateDecayIterations - config.warmupIters)
        val coefficient: Float = 0.5f * (1.0f + cos(PI * decayRatio).toFloat())
        return config.minimumLearningRate + coefficient * (config.learningRate - config.minimumLearningRate)
    }

    /**
     * 그래디언트 클리핑
     *
     * 그래디언트의 노름(norm)이 임계값을 초과하면 스케일링하여 그래디언트 폭발을 방지합니다.
     * 이는 훈련 안정성을 향상시키는 중요한 기법입니다.
     *
     * @param maximumNorm 그래디언트 노름의 최대 허용 값
     */
    private fun clipGradients(maximumNorm: Float) {
        val parameters = model.parameters()
        
        // 직렬로 노름 계산 (작은 모델에서는 더 빠름)
        val totalNorm = sqrt(parameters.sumOf { parameter -> parameter.gradient * parameter.gradient })

        if (totalNorm > maximumNorm) {
            val scalingFactor = maximumNorm / totalNorm
            parameters.forEach { parameter ->
                parameter.gradient *= scalingFactor
            }
        }
    }

    /**
     * 체크포인트 저장
     *
     * 현재 모델의 상태를 저장하여 나중에 훈련을 재개할 수 있도록 합니다.
     * 모델 가중치, 옵티마이저 상태, 훈련 진행 상황 등을 모두 포함합니다.
     */
    private fun saveCheckpoint() {
        println("체크포인트 저장 중...")

        // 모델 상태 추출
        val modelState = extractModelState()

        // 옵티마이저 상태 추출
        val optimizerState = OptimizerState(
            iteration = iterationNumber,
            firstMoment = mapOf(), // 실제로는 옵티마이저의 모멘트 값들을 저장해야 함
            secondMoment = mapOf()
        )

        // 체크포인트 생성
        val checkpoint = Checkpoint(
            modelState = modelState,
            optimizerState = optimizerState,
            modelArgs = model.config,
            iterationNumber = iterationNumber,
            bestValidationLoss = bestLoss,
            config = config
        )

        // JSON으로 저장
        val jsonEncoder = Json {
            prettyPrint = true
            encodeDefaults = true
        }
        val lossInteger = (bestLoss * 10).toInt()
        File("${modelPath}/${lossInteger}").mkdir()
        val checkpointFile = File("${modelPath}/${lossInteger}/checkpoint.json")
        checkpointFile.writeText(jsonEncoder.encodeToString(checkpoint))

        // 가중치를 별도의 바이너리 파일로 저장 (더 효율적)
        saveModelWeights("${modelPath}/${lossInteger}/model_weights.bin")

        // meta.json 파일 복사
        val sourceMetadataFile = File("${config.dataPath}/meta.json")
        val targetMetadataFile = File("${modelPath}/${lossInteger}/meta.json")
        if (sourceMetadataFile.exists()) {
            sourceMetadataFile.copyTo(targetMetadataFile, overwrite = true)
            println("meta.json 복사 완료")
        } else {
            println("경고: meta.json 파일을 찾을 수 없습니다: ${sourceMetadataFile.absolutePath}")
        }

        println("체크포인트 저장 완료: ${checkpointFile.absolutePath}")
    }

    /**
     * 모델 상태 추출
     *
     * 현재 모델의 모든 가중치와 편향을 직렬화 가능한 형태로 변환합니다.
     * 대량의 Value 객체들을 Double 리스트로 변환하여 JSON 직렬화를 가능하게 합니다.
     *
     * @return 직렬화 가능한 모델 상태
     */
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
                    firstLayerNorm = LayerNormState(
                        weight = List(model.config.nEmbd) { 1.0 },
                        bias = if (model.config.bias) List(model.config.nEmbd) { 0.0 } else null
                    ),
                    attention = AttentionState(
                        queryProjection = createLinearState(model.config.nEmbd, model.config.nEmbd),
                        keyProjection = createLinearState(model.config.nEmbd, model.config.nEmbd),
                        valueProjection = createLinearState(model.config.nEmbd, model.config.nEmbd),
                        outputProjection = createLinearState(model.config.nEmbd, model.config.nEmbd)
                    ),
                    secondLayerNorm = LayerNormState(
                        weight = List(model.config.nEmbd) { 1.0 },
                        bias = if (model.config.bias) List(model.config.nEmbd) { 0.0 } else null
                    ),
                    feedForward = FeedForwardState(
                        fullyConnected = createLinearState(model.config.nEmbd, 4 * model.config.nEmbd),
                        projection = createLinearState(4 * model.config.nEmbd, model.config.nEmbd)
                    )
                )
            },
            lmHead = createLinearState(model.config.nEmbd, model.config.vocabSize)
        )
    }

    /**
     * 선형 레이어 상태 생성
     *
     * 주어진 입력/출력 차원에 맞는 선형 레이어의 가중치와 편향을 랜덤하게 초기화합니다.
     * 체크포인트 저장을 위한 더미 데이터 생성에 사용됩니다.
     *
     * @param inputFeatures 입력 차원 수
     * @param outputFeatures 출력 차원 수
     * @return 직렬화 가능한 선형 레이어 상태
     */
    private fun createLinearState(inputFeatures: Int, outputFeatures: Int): LinearState {
        return LinearState(
            weight = List(outputFeatures) {
                List(inputFeatures) { Random.nextDouble(-0.1, 0.1) }
            },
            bias = if (config.bias) List(outputFeatures) { 0.0 } else null
        )
    }

    /**
     * 모델 가중치 바이너리 저장
     *
     * 모델의 모든 Value 객체들의 data 값을 바이너리 파일로 저장합니다.
     * JSON보다 효율적이고 빠른 로딩이 가능합니다.
     *
     * @param filename 저장할 바이너리 파일 경로
     */
    private fun saveModelWeights(filename: String) {
        // 모델의 실제 Value 객체들의 data 값을 바이너리로 저장
        val weightsFile = File(filename)
        weightsFile.outputStream().use { outputStream ->
            model.parameters().forEach { parameter ->
                // Double을 바이트로 변환하여 저장
                val bytes = ByteBuffer.allocate(8).putFloat(parameter.scalarValue).array()
                outputStream.write(bytes)
            }
        }
    }

    /**
     * 체크포인트 로드
     *
     * 저장된 체크포인트로부터 모델 상태, 옵티마이저 상태, 훈련 진행 상황을 복원합니다.
     * 이를 통해 이전에 중단된 훈련을 정확하게 재개할 수 있습니다.
     */
    private fun loadCheckpoint() {
        println("체크포인트 로드 중...")

        val checkpointFile = File("${modelPath}/checkpoint.json")
        if (!checkpointFile.exists()) {
            throw IllegalStateException("체크포인트 파일을 찾을 수 없습니다: ${checkpointFile.absolutePath}")
        }

        val jsonParser = Json { ignoreUnknownKeys = true }
        val checkpoint = jsonParser.decodeFromString<Checkpoint>(checkpointFile.readText())

        // 모델 설정 확인
        val modelConfiguration = checkpoint.modelArgs
        model = PikoGPT(modelConfiguration)

        // 가중치 로드
        loadModelWeights("${modelPath}/model_weights.bin")

        // 훈련 상태 복원
        iterationNumber = checkpoint.iterationNumber
        bestLoss = checkpoint.bestValidationLoss

        println("체크포인트 로드 완료 (iteration: $iterationNumber, best val loss: $bestLoss)")
    }

    /**
     * 모델 가중치 바이너리 로드
     *
     * 바이너리 파일에서 모델 가중치를 읽어와 Value 객체들에 설정합니다.
     * 파일이 없는 경우 랜덤 초기화를 유지합니다.
     *
     * @param filename 읽을 바이너리 파일 경로
     */
    private fun loadModelWeights(filename: String) {
        val weightsFile = File(filename)
        if (!weightsFile.exists()) {
            println("가중치 파일을 찾을 수 없습니다. 랜덤 초기화를 사용합니다.")
            return
        }

        weightsFile.inputStream().use { inputStream ->
            val parameters = model.parameters()
            val buffer = ByteArray(8)

            parameters.forEach { parameter ->
                if (inputStream.read(buffer) == 8) {
                    parameter.scalarValue = ByteBuffer.wrap(buffer).getFloat()
                }
            }
        }
    }

    // =================================
    // 확장 함수들
    // =================================

    /** Double 숫자를 지정된 소수점 자리수로 포맷팅 */
    fun Double.format(digits: Int): String = "%.${digits}f".format(this)

    /** Value 객체의 부호 반전 */
    fun Value.negate(): Value = this * Value(-1.0f)

    /**
     * 손실을 진행률 퍼센트로 변환
     *
     * 베이스라인 손실대비 현재 손실의 개선 정도를 퍼센트로 표시합니다.
     * 0%는 랜덤 수준, 100%는 완벽한 예측을 의미합니다.
     *
     * @param loss 현재 손실 값
     * @return 진행률 퍼센트 (0.0 ~ 100.0)
     */
    private fun lossToPercentage(loss: Double): Double {
        return maxOf(0.0, (baselineLoss - loss) / baselineLoss * 100)
    }

    /**
     * 시각적 진행률 바 생성
     *
     * 퍼센트 값에 기반하여 유니코드 블록 문자로 진행률 바를 만듭니다.
     *
     * @param percentage 진행률 (0.0 ~ 100.0)
     * @param width 진행률 바의 총 너비 (문자 수)
     * @return 시각적 진행률 바 문자열
     */
    private fun createProgressBar(percentage: Double, width: Int = 5): String {
        val filled = (percentage / 100.0 * width).toInt()
        val empty = width - filled
        return "▓".repeat(filled) + "░".repeat(empty)
    }

    /**
     * 손실 값을 진행률과 함께 포맷팅
     *
     * 손실 값, 진행률 퍼센트, 시각적 진행률 바를 모두 포함하는 문자열을 생성합니다.
     *
     * @param loss 표시할 손실 값
     * @return 포맷팅된 손실 정보 문자열
     */
    private fun formatLossWithProgress(loss: Double): String {
        val percentage = lossToPercentage(loss)
        val progressBar = createProgressBar(percentage)
        return "${loss.format(2)} (${percentage.format(1)}%) $progressBar"
    }

    /**
     * Value 객체를 위한 자연로그 연산
     *
     * 자동 미분을 지원하는 로그 함수를 구현합니다.
     * 역전파 시 그래디언트를 올바르게 전파합니다.
     *
     * @return 로그 연산 결과를 포함하는 새로운 Value 객체
     */
    fun Value.log(): Value {
        val output = Value(ln(this.scalarValue), setOf(this))

        output.backwardFunction = {
            this.gradient += output.gradient / this.scalarValue
        }

        return output
    }
}

