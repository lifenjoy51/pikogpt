package train

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.random.Random

// 데이터 로더
class DataLoader(
    private val dataPath: String,
    private val batchSize: Int,
    private val blockSize: Int
) {
    private lateinit var data: IntArray

    init {
        loadData()
    }

    private fun loadData() {
        val file = File(dataPath)
        val bytes = file.readBytes()
        val buffer = ByteBuffer.wrap(bytes)
        buffer.order(ByteOrder.LITTLE_ENDIAN)

        data = IntArray(bytes.size / 2)
        for (i in data.indices) {
            data[i] = buffer.getShort().toInt() and 0xFFFF
        }
        println("데이터 로드 완료: ${data.size} 토큰")
    }

    fun getBatch(): Pair<Array<IntArray>, Array<IntArray>> {
        val x = Array(batchSize) { IntArray(blockSize) }
        val y = Array(batchSize) { IntArray(blockSize) }

        for (i in 0 until batchSize) {
            val start = Random.nextInt(0, data.size - blockSize - 1)
            for (j in 0 until blockSize) {
                x[i][j] = data[start + j]
                y[i][j] = data[start + j + 1]
            }
        }

        return Pair(x, y)
    }
}