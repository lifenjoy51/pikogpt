package sample

import GradContext
import Value
import data.MetaInfo
import data.CharBPE
import data.CharBPE.Companion.UNKNOWN_TOKEN
import gpt.ScalarPikoGPT
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.withContext
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import train.ScalarCheckpoint
import java.io.File
import java.nio.ByteBuffer
import java.security.MessageDigest
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
class ScalarSampler(private val samplingConfiguration: SampleConfig) {
    /** 텍스트 생성에 사용될 GPT 모델 인스턴스 */
    private lateinit var textGenerationModel: ScalarPikoGPT

    /** 문자열을 토큰 ID 리스트로 변환하는 인코더 함수 */
    private lateinit var textToTokenEncoder: (String) -> List<Int>

    /** 토큰 ID 리스트를 문자열로 변환하는 디코더 함수 */
    private lateinit var tokenToTextDecoder: (List<Int>) -> String

    /** 모델의 어휘 사전 크기 (가능한 총 토큰 수) */
    private var vocabularySize: Int = 0

    /** 어휘 메타데이터 (디버깅용) */
    private lateinit var vocabularyMetadata: MetaInfo

    /** 이 샘플러 인스턴스의 고유 식별자 (로깅 및 추적용) */
    val uniqueIdentifier = MessageDigest.getInstance("MD5")
        .digest(samplingConfiguration.modelDirectoryPath.toByteArray())
        .joinToString("") { "%02x".format(it) }
        .take(4)

    /**
     * 샘플링에 사용되는 난수 생성기.
     * `samplingConfiguration.seed`로 시드된 독립 인스턴스 — `Random.Default`와 분리되어
     * 같은 시드로 항상 같은 시퀀스를 재현한다.
     */
    private val rng: Random = Random(samplingConfiguration.randomSeed)

    /**
     * 샘플러 초기화
     *
     * 모델 로드, 인코딩 설정을 순차적으로 수행합니다.
     */
    init {
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
        println("# 텍스트 생성 시작 #id:${uniqueIdentifier}")

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
        println("# 프롬프트 처리 완료 #id:${uniqueIdentifier} (입력 길이: ${initialText.length}자, 토큰 수: ${initialTokenIds.size}개)")

        // 여러 샘플을 병렬로 생성
        val generatedTexts = (0 until samplingConfiguration.numberOfSamples).map { sampleIndex ->
            async {
                println("# 샘플 생성 시작 #id:${uniqueIdentifier} (샘플 ${sampleIndex + 1}/${samplingConfiguration.numberOfSamples})")

                val generatedTokenIds = generateTokenSequence(
                    contextTokenIds = initialTokenIds,
                    maxNewTokens = samplingConfiguration.maximumNewTokens,
                    temperature = samplingConfiguration.samplingTemperature,
                    topKSize = samplingConfiguration.topKFilteringSize
                ).takeWhile { tokenId -> tokenId != 0 } // EOS 토큰 제거

                // 토큰 ID를 다시 텍스트로 변환
                val generatedText = tokenToTextDecoder(generatedTokenIds)
                println("# 샘플 생성 완료 #id:${uniqueIdentifier} (샘플 ${sampleIndex + 1}, 생성된 토큰 수: ${generatedTokenIds.size - initialTokenIds.size}개) 생성됨: $generatedText")

                generatedText
            }
        }.awaitAll()

        println("# 텍스트 생성 완료 #id:${uniqueIdentifier} (총 ${samplingConfiguration.numberOfSamples}개 샘플 생성)")

        GenerationResult(
            uniqueId = uniqueIdentifier,
            originalPrompt = initialText,
            generatedTexts = generatedTexts
        )
    }

    /**
     * 학습된 모델 로드
     *
     * 체크포인트 파일에서 모델 설정과 가중치를 로드하여 생성 준비를 완료합니다.
     * 오직 'resume' 모드만 지원하며, 'scratch' 모드는 학습된 모델이 필요하므로 오류를 발생시킵니다.
     */
    private fun loadTrainedModel() {
        when (samplingConfiguration.modelInitializationMode) {
            "resume" -> {
                val checkpointFile = File("${samplingConfiguration.modelDirectoryPath}/checkpoint.json")
                if (!checkpointFile.exists()) {
                    throw IllegalStateException("체크포인트 파일을 찾을 수 없습니다: ${checkpointFile.absolutePath}")
                }

                // 체크포인트 데이터 파싱
                val jsonParser = Json { ignoreUnknownKeys = true }
                val checkpointData = jsonParser.decodeFromString<ScalarCheckpoint>(checkpointFile.readText())

                // 모델 아키텍처 구성 및 인스턴스 생성
                val modelArchitectureConfig = checkpointData.modelArgs
                textGenerationModel = ScalarPikoGPT(modelArchitectureConfig)
                vocabularySize = modelArchitectureConfig.vocabularySize

                // 모델 가중치 로드
                loadModelWeights("${samplingConfiguration.modelDirectoryPath}/model_weights.bin")

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
            val byteBuffer = ByteArray(4)

            // 모든 파라미터에 대해 순차적으로 가중치 로드
            modelParameters.forEach { parameter ->
                if (inputStream.read(byteBuffer) == 4) {
                    // 4바이트 Float 값을 직접 설정
                    parameter.scalarValue = ByteBuffer.wrap(byteBuffer).float
                }
            }
        }
    }

    /**
     * 토큰화 인코딩/디코딩 설정.
     *
     * `meta.json`에 **BPE merges가 포함되어 있으면** 학습 때와 완전히 동일한
     * `CharBPE.encode` 경로를 복원해 사용한다. merges가 비어 있으면
     * (구버전 체크포인트) 그리디 최장매칭으로 폴백.
     */
    private fun setupTokenization() {
        val metadataFile = File("${samplingConfiguration.modelDirectoryPath}/meta.json")
        val jsonParser = Json { ignoreUnknownKeys = true }
        vocabularyMetadata = jsonParser.decodeFromString<MetaInfo>(metadataFile.readText())

        if (vocabularyMetadata.merges.isNotEmpty()) {
            // BPE 복원 — 학습 시와 동일한 플래그로 인스턴스 생성 후 stoi + merges 주입
            val bpe = CharBPE(
                maxVocabSize = vocabularyMetadata.vocabularySize,
                specialTokens = vocabularyMetadata.specialTokens,
                lowercase = vocabularyMetadata.lowercase,
                useWordPreTokenize = vocabularyMetadata.useWordPreTokenize,
                standardBpeScoring = true,
                verbose = false,
            )
            val mergePairs = vocabularyMetadata.merges.map { it[0] to it[1] }
            bpe.restore(vocabularyMetadata.stringToIndex, mergePairs)
            textToTokenEncoder = { inputText -> bpe.encode(inputText) }
        } else {
            // 구버전 meta.json → 그리디 폴백
            textToTokenEncoder = { inputText ->
                greedyTokenize(inputText, vocabularyMetadata.stringToIndex)
            }
        }

        // 디코더: 토큰 ID → 문자열. 모든 specialTokens(예: <|bos|>, <|eos|>, <|unk|>)은 출력에서 제외.
        val specialTokenSet = vocabularyMetadata.specialTokens.toHashSet()
        tokenToTextDecoder = { tokenIdList ->
            tokenIdList
                .mapNotNull { id -> vocabularyMetadata.indexToString[id]?.takeUnless { it in specialTokenSet } }
                .joinToString("")
        }

        vocabularySize = vocabularyMetadata.vocabularySize
    }

    /**
     * 그리디 최장매칭 토큰화
     * 
     * 문자열에서 가장 긴 매칭되는 토큰을 찾아 토큰화합니다.
     * BPE 병합 규칙이 없을 때 사용할 수 있는 대안적 접근법입니다.
     * 
     * @param text 토큰화할 텍스트
     * @param stoi 문자열-ID 매핑
     * @return 토큰 ID 리스트
     */
    private fun greedyTokenize(text: String, stoi: Map<String, Int>): List<Int> {
        val tokens = mutableListOf<Int>()
        var i = 0
        val longestTokenLength = stoi.keys.maxOf { it.length }
        
        while (i < text.length) {
            var found = false
            
            // 현재 위치에서 가장 긴 매칭 토큰 찾기
            for (length in minOf(text.length - i, longestTokenLength) downTo 1) {
                val candidate = text.substring(i, i + length)
                val tokenId = stoi[candidate]
                
                if (tokenId != null) {
                    tokens.add(tokenId)
                    i += length
                    found = true
                    break
                }
            }
            
            // 매칭되는 토큰이 없으면 UNK 토큰으로 처리하고 한 글자씩 진행
            if (!found) {
                tokens.add(1) // UNK 토큰 ID
                i++
            }
        }
        
        return tokens
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
    ): List<Int> = GradContext.noGrad {
        val generatedSequence = contextTokenIds.toMutableList()
        var currentContext = contextTokenIds.toIntArray()

        repeat(maxNewTokens) { stepIndex ->
            // 진행 상황 로깅 (10% 간격으로)

            //val progressPercent = (stepIndex * 100) / maxNewTokens
            //println("# 토큰 생성 진행 #id:${uniqueIdentifier} ${progressPercent}% 완료 ${stepIndex}/${maxNewTokens} 토큰")


            // 컨텍스트 길이가 모델의 최대 블록 크기를 초과하면 자르기
            if (currentContext.size > textGenerationModel.config.maxSequenceLength) {
                currentContext = currentContext.takeLast(textGenerationModel.config.maxSequenceLength).toIntArray()
            }

            // 모델을 사용하여 다음 토팠 예측
            val outputLogits = textGenerationModel.forward(currentContext)
            val finalPositionLogits = outputLogits.getLastPositionLogits() // 마지막 위치의 로짓 (다음 토팠 예측용)

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
            val logitsData = arrayOf(filteredLogits)
            val logits = gpt.Logits(logitsData)
            val softmaxResult = logits.softmax()
            val tokenProbabilities = softmaxResult.get(0)
                .map { it.scalarValue }.toFloatArray()

            // === 디버깅 정보 출력 ===
            if (stepIndex < 10) { // 처음 10 스텝만 출력
                println("Step $stepIndex:")
                
                // 1. Top-5 토큰과 확률 출력
                val top5 = tokenProbabilities.withIndex()
                    .sortedByDescending { it.value }
                    .take(5)
                
                println("  Top-5 tokens:")
                top5.forEach { (tokenId, prob) ->
                    val tokenText = vocabularyMetadata.indexToString[tokenId] ?: "?"
                    println("    ID:$tokenId '${tokenText.replace("\n", "\\n").replace(UNKNOWN_TOKEN, "[FULL_SPACE]")}' prob:${prob.format(4)}")
                }
                
                // 2. Full-width space 토큰(ID 1)의 확률 확인
                val fullSpaceProb = tokenProbabilities.getOrNull(1) ?: 0.0f
                println("  Full-width space (ID:1) probability: ${fullSpaceProb.format(4)}")
                
                // 3. 현재 컨텍스트 출력
                val contextText = tokenToTextDecoder(currentContext.toList())
                println("  Current context: '${contextText.takeLast(20)}'")
            }

            val selectedToken = sampleFromDistribution(tokenProbabilities)
            
            // 선택된 토큰 정보 출력
            if (stepIndex < 10) {
                val selectedTokenText = vocabularyMetadata.indexToString[selectedToken] ?: "?"
                val selectedProb = tokenProbabilities.getOrNull(selectedToken) ?: 0.0f
                println("  Selected: ID:$selectedToken '${selectedTokenText.replace(UNKNOWN_TOKEN, "[FULL_SPACE]")}' prob:${selectedProb.format(4)}")
                println()
            }

            // 생성된 토팠을 시퀀스에 추가
            generatedSequence.add(selectedToken)
            currentContext = currentContext + selectedToken
        }

        generatedSequence
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
     * 확률 분포에서 토팠 샘플링
     *
     * 주어진 확률 분포에 따라 랜덤하게 토팠을 선택합니다.
     * 누적 분포 함수(CDF)를 사용하여 효율적으로 샘플링합니다.
     *
     * @param probabilityDistribution 토팠별 확률 분포 배열
     * @return 선택된 토팠의 인덱스
     */
    private fun sampleFromDistribution(probabilityDistribution: FloatArray): Int {
        // FIXME 로깅 상세하게 추가.
        // 누적 확률 분포 함수(CDF) 계산
        val cumulativeProbabilities = FloatArray(probabilityDistribution.size)
        cumulativeProbabilities[0] = probabilityDistribution[0]

        for (tokenIndex in 1 until probabilityDistribution.size) {
            cumulativeProbabilities[tokenIndex] = cumulativeProbabilities[tokenIndex - 1] + probabilityDistribution[tokenIndex]
        }

        // 0.0과 1.0 사이의 랜덤 값 생성 (시드된 rng 사용 → 재현성 보장)
        val randomValue = rng.nextDouble()

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

    // Float 포맷팅을 위한 확장 함수
    private fun Float.format(decimals: Int): String = "%.${decimals}f".format(this)
}