package train

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.random.Random

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
) {
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
        println("데이터 로드 완룮: ${tokenData.size} 토큰")
    }

    /**
     * 훈련/검증을 위한 미니배치 생성
     *
     * 랜덤한 시작 위치에서 batchSize만큼의 시퀀스를 생성합니다.
     * 각 시퀀스는 blockSize 길이를 가지며, 입력과 타곟 시퀀스는 1토큰씩 시프트됩니다.
     *
     * 예시:
     * - 원본 데이터: [1, 2, 3, 4, 5, 6, 7, ...]
     * - 입력 시퀀스: [1, 2, 3, 4, 5]
     * - 타곟 시퀀스: [2, 3, 4, 5, 6]
     *
     * @return Pair<입력_시퀀스, 타곟_시퀀스> 형태의 배치 데이터
     */
    fun getBatch(): Pair<Array<IntArray>, Array<IntArray>> {
        val inputSequences = Array(batchSize) { IntArray(blockSize) }
        val targetSequences = Array(batchSize) { IntArray(blockSize) }

        for (batchIndex in 0 until batchSize) {
            // 랜덤 시작 위치 선택 (마지막에 blockSize+1만큼 여유 공간 필요)
            val startPosition = Random.nextInt(0, tokenData.size - blockSize - 1)

            for (sequenceIndex in 0 until blockSize) {
                // 입력: [startPosition, startPosition+blockSize-1]
                inputSequences[batchIndex][sequenceIndex] = tokenData[startPosition + sequenceIndex]
                // 타곟: [startPosition+1, startPosition+blockSize] (1토큰 시프트)
                targetSequences[batchIndex][sequenceIndex] = tokenData[startPosition + sequenceIndex + 1]
            }
        }

        return Pair(inputSequences, targetSequences)
    }
}