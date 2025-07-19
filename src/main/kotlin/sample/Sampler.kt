package sample

import Value
import data.MetaInfo
import gpt.PikoGPT
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.withContext
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import train.Checkpoint
import java.io.File
import java.nio.ByteBuffer
import java.util.*
import kotlin.math.exp
import kotlin.random.Random

/**
 * GPT 모델 텍스트 생성기
 *
 * 학습된 GPT 모델을 사용하여 주어진 프롬프트에서 새로운 텍스트를 생성합니다.
 * 다양한 샘플링 전략(온도, Top-K)을 지원하여 창의적이고 연관성 있는 텍스트를 생성합니다.
 *
 * 주요 기능:
 * - 체크포인트에서 모델 로드
 * - 다양한 샘플링 전략 지원
 * - 병렬 텍스트 생성
 * - 자동 인코딩/디코딩
 *
 * @param samplingConfiguration 샘플링 설정 (온도, Top-K, 랜덤 시드 등)
 */
class Sampler(private val samplingConfiguration: SampleConfig) {
    /** 텍스트 생성에 사용될 GPT 모델 인스턴스 */
    private lateinit var textGenerationModel: PikoGPT

    /** 문자열을 토큰 ID 리스트로 변환하는 인코더 함수 */
    private lateinit var textToTokenEncoder: (String) -> List<Int>

    /** 토큰 ID 리스트를 문자열로 변환하는 디코더 함수 */
    private lateinit var tokenToTextDecoder: (List<Int>) -> String

    /** 모델의 어휘 사전 크기 (가능한 총 토큰 수) */
    private var vocabularySize: Int = 0

    /** 이 샘플러 인스턴스의 고유 식별자 (로깅 및 추적용) */
    val uniqueIdentifier = UUID.randomUUID().toString()

    /**
     * 샘플러 초기화
     *
     * 랜덤 시드 설정, 모델 로드, 인코딩 설정을 순차적으로 수행합니다.
     */
    init {
        // 재현 가능한 결과를 위한 랜덤 시드 설정
        Random(samplingConfiguration.seed)

        // 학습된 모델 로드
        loadTrainedModel()

        // 토큰화 인코더/디코더 설정
        setupTokenization()
    }

    /**
     * 주어진 프롬프트에서 텍스트 생성
     *
     * 비동기적으로 여러 샘플을 병렬로 생성하여 다양성을 높입니다.
     * 각 샘플은 동일한 설정을 사용하지만 랜덤 요소로 인해 서로 다른 결과를 생성합니다.
     *
     * @param inputPrompt 생성을 시작할 초기 텍스트 ('FILE:' 접두사로 파일 경로 지정 가능)
     * @return 생성 결과를 담은 Result 객체 (UID, 프롬프트, 생성된 텍스트 리스트 포함)
     */
    suspend fun generateText(inputPrompt: String): GenerationResult = withContext(Dispatchers.Default) {
        // 입력 프롬프트 처리 (파일에서 읽기 또는 직접 사용)
        val initialText = if (inputPrompt.startsWith("FILE:")) {
            // 파일에서 텍스트 로드
            File(inputPrompt.substring(5)).readText()
        } else {
            // 직접 입력된 텍스트 사용
            inputPrompt
        }

        // 시작 텍스트를 토큰 ID로 변환
        val initialTokenIds = textToTokenEncoder(initialText)

        // 여러 샘플을 병렬로 생성
        val generatedTexts = (0 until samplingConfiguration.numSamples).map { sampleIndex ->
            async {
                val generatedTokenIds = generateTokenSequence(
                    contextTokenIds = initialTokenIds,
                    maxNewTokens = samplingConfiguration.maxNewTokens,
                    temperature = samplingConfiguration.temperature,
                    topKSize = samplingConfiguration.topK
                ).takeWhile { tokenId -> tokenId != 0 } // EOS 토큰 제거

                // 토큰 ID를 다시 텍스트로 변환
                tokenToTextDecoder(generatedTokenIds)
            }
        }.awaitAll()

        GenerationResult(
            uniqueId = uniqueIdentifier,
            originalPrompt = initialText,
            generatedTexts = generatedTexts
        )
    }

    /**
     * 호환성을 위한 별칭 메서드
     *
     * @param prompt 생성을 시작할 초기 텍스트
     * @return 생성 결과를 담은 Result 객체
     */
    suspend fun sample(prompt: String): GenerationResult = generateText(prompt)

    /**
     * 학습된 모델 로드
     *
     * 체크포인트 파일에서 모델 설정과 가중치를 로드하여 생성 준비를 완료합니다.
     * 오직 'resume' 모드만 지원하며, 'scratch' 모드는 학습된 모델이 필요하므로 오류를 발생시킵니다.
     */
    private fun loadTrainedModel() {
        when (samplingConfiguration.initFrom) {
            "resume" -> {
                val checkpointFile = File("${samplingConfiguration.modelDir}/checkpoint.json")
                if (!checkpointFile.exists()) {
                    throw IllegalStateException("체크포인트 파일을 찾을 수 없습니다: ${checkpointFile.absolutePath}")
                }

                // 체크포인트 데이터 파싱
                val jsonParser = Json { ignoreUnknownKeys = true }
                val checkpointData = jsonParser.decodeFromString<Checkpoint>(checkpointFile.readText())

                // 모델 아키텍처 구성 및 인스턴스 생성
                val modelArchitectureConfig = checkpointData.modelArgs
                textGenerationModel = PikoGPT(modelArchitectureConfig)
                vocabularySize = modelArchitectureConfig.vocabSize

                // 모델 가중치 로드
                loadModelWeights("${samplingConfiguration.modelDir}/model_weights.bin")

                println("# 모델 로드 완료 #id:${uniqueIdentifier} (iteration: ${checkpointData.iterationNumber}, val loss: ${checkpointData.bestValidationLoss})")
            }

            "scratch" -> {
                throw IllegalArgumentException("텍스트 생성을 위해서는 사전 학습된 모델이 필수입니다. 'resume' 모드를 사용하세요.")
            }
        }
    }

    /**
     * 모델 가중치 로드
     *
     * 바이너리 파일에서 모델의 모든 파라미터 가중치를 순차적으로 읽어와 설정합니다.
     * 파일이 없는 경우 경고 메시지를 출력하고 랜덤 가중치를 유지합니다.
     *
     * @param weightsFilePath 가중치 바이너리 파일의 경로
     */
    private fun loadModelWeights(weightsFilePath: String) {
        val weightsFile = File(weightsFilePath)
        if (!weightsFile.exists()) {
            println("경고: 가중치 파일을 찾을 수 없습니다 ($weightsFilePath). 랜덤 가중치를 사용합니다.")
            return
        }

        weightsFile.inputStream().use { inputStream ->
            val modelParameters = textGenerationModel.parameters()
            val byteBuffer = ByteArray(8)

            // 모든 파라미터에 대해 순차적으로 가중치 로드
            modelParameters.forEach { parameter ->
                if (inputStream.read(byteBuffer) == 8) {
                    // 8바이트 Double 값을 Float으로 변환하여 설정
                    parameter.scalarValue = ByteBuffer.wrap(byteBuffer).double.toFloat()
                }
            }
        }
    }

    /**
     * 토큰화 인코딩/디코딩 설정
     *
     * meta.json 파일에서 어휘 매핑 정보를 로드하여 인코더와 디코더 함수를 설정합니다.
     * 이 함수들은 텍스트와 토큰 ID 간의 상호 변환에 사용됩니다.
     */
    private fun setupTokenization() {
        // 어휘 메타데이터 파일 로드
        val metadataFile = File("${samplingConfiguration.modelDir}/meta.json")
        val jsonParser = Json { ignoreUnknownKeys = true }
        val vocabularyMetadata = jsonParser.decodeFromString<MetaInfo>(metadataFile.readText())

        // 인코더 함수: 문자열 → 토큰 ID 리스트
        textToTokenEncoder = { inputText ->
            inputText.map { character ->
                vocabularyMetadata.stoi[character.toString()] ?: 1  // 알 수 없는 문자는 UNK 토큰(1)로 처리
            }
        }

        // 디코더 함수: 토큰 ID 리스트 → 문자열
        tokenToTextDecoder = { tokenIdList ->
            tokenIdList.joinToString("") { tokenId ->
                vocabularyMetadata.itos[tokenId] ?: " "  // 알 수 없는 토큰 ID는 공백으로 처리
            }
        }

        vocabularySize = vocabularyMetadata.vocabSize
    }

    /**
     * 토큰 시퀀스 생성
     *
     * 주어진 컨텍스트에서 시작하여 새로운 토큰들을 순차적으로 생성합니다.
     * 각 스텝에서 모델이 다음 토큰을 예측하고, 샘플링 전략에 따라 선택합니다.
     *
     * @param contextTokenIds 생성을 시작할 컨텍스트 토큰 ID 리스트
     * @param maxNewTokens 생성할 최대 새 토큰 수
     * @param temperature 샘플링 온도 (낮을수록 결정론적, 높을수록 창의적)
     * @param topKSize Top-K 필터링 크기 (가장 가능성 높은 K개 토큰만 고려)
     * @return 생성된 전체 토큰 시퀀스 (컨텍스트 + 새 토큰들)
     */
    private fun generateTokenSequence(
        contextTokenIds: List<Int>,
        maxNewTokens: Int,
        temperature: Float,
        topKSize: Int
    ): List<Int> {
        val generatedSequence = contextTokenIds.toMutableList()
        var currentContext = contextTokenIds.toIntArray()

        repeat(maxNewTokens) {
            // 컨텍스트 길이가 모델의 최대 블록 크기를 초과하면 자르기
            if (currentContext.size > textGenerationModel.config.blockSize) {
                currentContext = currentContext.takeLast(textGenerationModel.config.blockSize).toIntArray()
            }

            // 모델을 사용하여 다음 토팠 예측
            val outputLogits = textGenerationModel.forward(currentContext)
            val finalPositionLogits = outputLogits.last() // 마지막 위치의 로짓 (다음 토팠 예측용)

            // 온도 스케일링 적용 (높은 온도는 더 다양한 선택)
            val temperatureScaledLogits = finalPositionLogits.map { logitValue ->
                Value(logitValue.scalarValue / temperature)
            }.toTypedArray()

            // Top-K 필터링 적용 (가장 가능성 높은 K개만 유지)
            val filteredLogits = if (topKSize > 0 && topKSize < vocabularySize) {
                applyTopKFiltering(temperatureScaledLogits, topKSize)
            } else {
                temperatureScaledLogits
            }

            // Softmax 확률 분포 계산 및 토팠 샘플링
            val tokenProbabilities = softmax(filteredLogits)
            val selectedToken = sampleFromDistribution(tokenProbabilities)

            // 생성된 토팠을 시퀀스에 추가
            generatedSequence.add(selectedToken)
            currentContext = currentContext + selectedToken
        }

        return generatedSequence
    }

    /**
     * Top-K 필터링 적용
     *
     * 로짓 배열에서 가장 높은 K개의 값만 유지하고 나머지는 매우 작은 값으로 마스킹합니다.
     * 이를 통해 매우 낮은 확률의 토팠들을 샘플링에서 제외하여 더 안정적인 생성을 도모합니다.
     *
     * @param logitArray 필터링할 로짓 배열
     * @param topKCount 유지할 상위 토팠의 수
     * @return Top-K 필터링이 적용된 로짓 배열
     */
    private fun applyTopKFiltering(logitArray: Array<Value>, topKCount: Int): Array<Value> {
        // 로짓 값을 내림차순으로 정렬하여 상위 K개의 인덱스 추출
        val sortedIndicesWithValues = logitArray.withIndex().sortedByDescending { it.value.scalarValue }
        val topKIndices = sortedIndicesWithValues.take(topKCount).map { it.index }.toSet()

        // 상위 K개가 아닌 로짓은 -∞로 설정하여 Softmax에서 확률 0으로 만듦
        return logitArray.mapIndexed { tokenIndex, originalLogit ->
            if (tokenIndex in topKIndices) {
                originalLogit  // 상위 K개는 원래 값 유지
            } else {
                Value(Float.NEGATIVE_INFINITY)  // 나머지는 마스킹
            }
        }.toTypedArray()
    }

    /**
     * Softmax 확률 분포 계산
     *
     * 로짓 배열을 확률 분포로 변환합니다.
     * 수치 안정성을 위해 최대값을 미리 빼서 오버플로우를 방지합니다.
     *
     * @param logitArray 로짓 배열
     * @return 정규화된 확률 분포 (합이 1.0이 되는 배열)
     */
    private fun softmax(logitArray: Array<Value>): FloatArray {
        // 수치 안정성을 위해 최대 로짓 값을 미리 빼기
        val maximumLogit = logitArray.maxByOrNull { it.scalarValue }?.scalarValue ?: 0.0f

        // 지수 함수 적용 (max 값을 빼서 안정성 확보)
        val exponentialValues = logitArray.map { logit -> exp(logit.scalarValue - maximumLogit) }

        // 지수 값들의 합계 계산
        val exponentialSum = exponentialValues.sum()

        // 정규화하여 확률 분포 생성
        return exponentialValues.map { expValue -> expValue / exponentialSum }.toFloatArray()
    }

    /**
     * 확률 분포에서 토팠 샘플링
     *
     * 주어진 확률 분포에 따라 랜덤하게 토팠을 선택합니다.
     * 누적 분포 함수(CDF)를 사용하여 효율적으로 샘플링합니다.
     *
     * @param probabilityDistribution 토팠별 확률 분포 배열
     * @return 선택된 토팠의 인덱스
     */
    private fun sampleFromDistribution(probabilityDistribution: FloatArray): Int {
        // 누적 확률 분포 함수(CDF) 계산
        val cumulativeProbabilities = FloatArray(probabilityDistribution.size)
        cumulativeProbabilities[0] = probabilityDistribution[0]

        for (tokenIndex in 1 until probabilityDistribution.size) {
            cumulativeProbabilities[tokenIndex] = cumulativeProbabilities[tokenIndex - 1] + probabilityDistribution[tokenIndex]
        }

        // 0.0과 1.0 사이의 랜덤 값 생성
        val randomValue = Random.nextDouble()

        // CDF에서 랜덤 값보다 큰 최초 인덱스 찾기
        for (tokenIndex in cumulativeProbabilities.indices) {
            if (randomValue <= cumulativeProbabilities[tokenIndex]) {
                return tokenIndex
            }
        }

        // 안전장치: 모든 경우를 빠져나온 경우 마지막 토팠 반환
        return probabilityDistribution.size - 1
    }

    /**
     * 텍스트 생성 결과를 담는 데이터 클래스
     *
     * @param uniqueId 생성 세션의 고유 식별자
     * @param originalPrompt 생성에 사용된 원본 프롬프트
     * @param generatedTexts 생성된 모든 텍스트 샘플들의 리스트
     */
    data class GenerationResult(
        val uniqueId: String,
        val originalPrompt: String,
        val generatedTexts: List<String>
    ) {
        // 호환성을 위한 별칭 속성들
        val uid: String get() = uniqueId
        val prompt: String get() = originalPrompt
        val results: List<String> get() = generatedTexts
    }
}