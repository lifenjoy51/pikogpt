package train

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.random.Random

/**
 * 미니배치 공급자 인터페이스. 단일 코퍼스 [DataLoader]와 두 코퍼스 mix [MixedDataLoader]가 모두 구현.
 * VecTrainer.trainLoader 같은 호출 측이 다형적으로 사용할 수 있게 함.
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
 * 두 코퍼스를 시퀀스 단위로 섞어 미니배치를 만드는 로더.
 *
 * IT(finetune) 단계에서 BASE replay를 위해 사용. 각 미니배치 row마다
 * Bernoulli(p=[replayRatio])로 primary(IT) 또는 replay(BASE)에서 시퀀스를 추첨.
 * gradient accumulation 32 × batch 2 = 64 시퀀스/step에서 replay 0.2면 ~13개 replay.
 *
 * 두 underlying DataLoader는 각자 batchSize 분량의 batch를 만든 뒤, row별로 골라잡아
 * 합친다 — 호출 비용은 각 단일 DataLoader 대비 약 2× (한쪽 배치는 거의 사용 안 됨).
 * batch=2 수준이라 이 비용은 무시 가능.
 *
 * @param primaryPath  IT 데이터 .bin 경로
 * @param replayPath   BASE 데이터 .bin 경로
 * @param replayRatio  replay 비율 (0.0~1.0)
 * @param batchSize    한 batch의 시퀀스 수
 * @param blockSize    각 시퀀스 길이
 */
class MixedDataLoader(
    primaryPath: String,
    replayPath: String,
    private val replayRatio: Float,
    private val batchSize: Int,
    private val blockSize: Int,
) : BatchSource {
    private val primary = DataLoader(primaryPath, batchSize, blockSize)
    private val replay = DataLoader(replayPath, batchSize, blockSize)

    init {
        require(replayRatio in 0.0f..1.0f) { "replayRatio는 0~1 범위여야 함: $replayRatio" }
    }

    override fun getBatch(): Pair<Array<IntArray>, Array<IntArray>> {
        val (pIn, pTgt) = primary.getBatch()
        val (rIn, rTgt) = replay.getBatch()
        val inputSequences = Array(batchSize) { IntArray(blockSize) }
        val targetSequences = Array(batchSize) { IntArray(blockSize) }
        for (b in 0 until batchSize) {
            val useReplay = Random.nextFloat() < replayRatio
            val src = if (useReplay) Pair(rIn, rTgt) else Pair(pIn, pTgt)
            for (i in 0 until blockSize) {
                inputSequences[b][i] = src.first[b][i]
                targetSequences[b][i] = src.second[b][i]
            }
        }
        return Pair(inputSequences, targetSequences)
    }
}