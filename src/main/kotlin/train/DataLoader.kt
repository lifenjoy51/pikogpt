package train

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.random.Random

/**
 * 미니배치 공급자 인터페이스. scalar 기본 [DataLoader] + turbo 전용 변형들이 모두 구현.
 * Turbo 전용 변형(`RecordAware`, `ChunkAnchored`, `Mixed`, `Triple`)은 `turbo/TurboDataLoaders.kt`.
 */
interface BatchSource {
    fun getBatch(): Pair<Array<IntArray>, Array<IntArray>>
}

/**
 * 데이터 로더 클래스
 *
 * 전처리된 토큰화 데이터를 로드하고, 훈련/검증을 위한 미니배치를 생성합니다.
 * 데이터는 바이너리 형식(.bin)으로 저장된 정수 매열로 구성되어 있습니다.
 *
 * @param dataPath 데이터 파일의 경로 (.bin 파일)
 * @param batchSize 한 번에 처리할 시퀀스의 개수 (배치 크기)
 * @param blockSize 각 시퀀스의 최대 길이 (맥락 윈도우)
 */
class DataLoader(
    private val dataPath: String,
    private val batchSize: Int,
    private val blockSize: Int
) : BatchSource {
    /** 로드된 토큰 데이터 배열 */
    private lateinit var tokenData: IntArray

    /**
     * 데이터 로더 초기화
     * 생성자에서 자동으로 데이터를 로드합니다.
     */
    init {
        loadData()
    }

    /**
     * 바이너리 데이터 파일 로드
     *
     * .bin 파일에서 토큰 데이터를 읽어옵니다.
     * 각 토큰은 4바이트 정수로 저장되어 있으며, Big Endian 순서로 읽습니다.
     *
     * 데이터 형식:
     * - 각 토큰: 4바이트 정수
     * - 전체 파일: [token1][token2][token3]...
     */
    private fun loadData() {
        val file = File(dataPath)
        val bytes = file.readBytes()
        val buffer = ByteBuffer.wrap(bytes)
        buffer.order(ByteOrder.BIG_ENDIAN)

        // 4바이트씩 읽어서 정수 배열로 변환
        tokenData = IntArray(bytes.size / 4)
        for (tokenIndex in tokenData.indices) {
            tokenData[tokenIndex] = buffer.getInt()
        }
        println("데이터 로드 완료: ${tokenData.size} 토큰")
    }

    /**
     * 훈련/검증을 위한 미니배치 생성
     *
     * 랜덤한 시작 위치에서 batchSize만큼의 시퀀스를 생성합니다.
     * 각 시퀀스는 blockSize 길이를 가지며, 입력과 타겟 시퀀스는 1토큰씩 시프트됩니다.
     *
     * 예시:
     * - 원본 데이터: [1, 2, 3, 4, 5, 6, 7, ...]
     * - 입력 시퀀스: [1, 2, 3, 4, 5]
     * - 타겟 시퀀스: [2, 3, 4, 5, 6]
     *
     * @return Pair<입력_시퀀스, 타겟_시퀀스> 형태의 배치 데이터
     */
    override fun getBatch(): Pair<Array<IntArray>, Array<IntArray>> {
        val inputSequences = Array(batchSize) { IntArray(blockSize) }
        val targetSequences = Array(batchSize) { IntArray(blockSize) }

        for (batchIndex in 0 until batchSize) {
            // 랜덤 시작 위치 선택 (마지막에 blockSize+1만큼 여유 공간 필요)
            val margin = blockSize + 1
            val startPosition = Random.nextInt(0, tokenData.size - margin)

            for (sequenceIndex in 0 until blockSize) {
                // 입력: [startPosition, startPosition+blockSize-1]
                inputSequences[batchIndex][sequenceIndex] = tokenData[startPosition + sequenceIndex]
                // 타겟: [startPosition+1, startPosition+blockSize] (1토큰 시프트)
                targetSequences[batchIndex][sequenceIndex] = tokenData[startPosition + sequenceIndex + 1]
            }
        }

        return Pair(inputSequences, targetSequences)
    }
}

/**
 * 두 개 source(`primary`, `secondary`)를 가중 베르누이 sampling으로 섞는 DataLoader.
 *
 * batch 내 각 시퀀스는 독립적으로 secondary 확률 `secondaryProb`로 secondary stream에서,
 * 나머지 확률(1 - secondaryProb)로 primary stream에서 random offset chunk를 뽑는다.
 *
 * CCMC-all v2 사용: primary=other(stories/dialogues/wiki/cause_seq/chained/counting),
 * secondary=lemma_sentences, secondaryProb=0.1로 lemma chunk EOS 빈도 균형.
 *
 * @param primaryPath 메인 stream binary 경로 (예: train_other.bin)
 * @param secondaryPath 가중치 낮출 stream binary 경로 (예: train_lemma.bin)
 * @param secondaryProb 0.0~1.0, batch 시퀀스가 secondary에서 뽑힐 확률
 * @param batchSize 배치 크기
 * @param blockSize 시퀀스 길이
 */
class WeightedSourceDataLoader(
    private val primaryPath: String,
    private val secondaryPath: String,
    private val secondaryProb: Float,
    private val batchSize: Int,
    private val blockSize: Int,
) : BatchSource {
    private lateinit var primaryData: IntArray
    private lateinit var secondaryData: IntArray

    init {
        require(secondaryProb in 0.0f..1.0f) { "secondaryProb 범위는 [0,1], 받은 값: $secondaryProb" }
        primaryData = readBinary(primaryPath)
        println("Primary 데이터 로드 완료: ${primaryData.size} 토큰 ($primaryPath)")
        secondaryData = readBinary(secondaryPath)
        println("Secondary 데이터 로드 완료: ${secondaryData.size} 토큰 (secondaryProb=$secondaryProb, $secondaryPath)")
    }

    private fun readBinary(path: String): IntArray {
        val bytes = File(path).readBytes()
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN)
        val arr = IntArray(bytes.size / 4)
        for (i in arr.indices) arr[i] = buffer.getInt()
        return arr
    }

    override fun getBatch(): Pair<Array<IntArray>, Array<IntArray>> {
        val inputSequences = Array(batchSize) { IntArray(blockSize) }
        val targetSequences = Array(batchSize) { IntArray(blockSize) }
        val margin = blockSize + 1

        for (batchIndex in 0 until batchSize) {
            val useSecondary = Random.nextFloat() < secondaryProb
            val data = if (useSecondary) secondaryData else primaryData
            val startPosition = Random.nextInt(0, data.size - margin)
            for (sequenceIndex in 0 until blockSize) {
                inputSequences[batchIndex][sequenceIndex] = data[startPosition + sequenceIndex]
                targetSequences[batchIndex][sequenceIndex] = data[startPosition + sequenceIndex + 1]
            }
        }

        return Pair(inputSequences, targetSequences)
    }
}
