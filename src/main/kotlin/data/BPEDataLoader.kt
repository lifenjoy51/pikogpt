package data

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

// 데이터 로더
class BPEDataLoader {
    fun loadBinary(filename: String): IntArray {
        val file = File(filename)
        val bytes = file.readBytes()
        val buffer = ByteBuffer.wrap(bytes)
        buffer.order(ByteOrder.LITTLE_ENDIAN)

        val ids = IntArray(bytes.size / 2)
        for (i in ids.indices) {
            ids[i] = buffer.getShort().toInt() and 0xFFFF
        }

        return ids
    }
}