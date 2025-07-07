package data


import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder


// 데이터 로드를 위한 유틸리티 클래스
class DataLoader {
    fun loadBinary(filename: String): IntArray {
        val file = File(filename)
        val bytes = file.readBytes()
        val buffer = ByteBuffer.wrap(bytes)
        buffer.order(ByteOrder.LITTLE_ENDIAN)

        val ids = IntArray(bytes.size / 2)
        for (i in ids.indices) {
            ids[i] = buffer.getShort().toInt() and 0xFFFF // uint16으로 읽기
        }

        return ids
    }

    fun loadMetaInfo(filename: String): MetaInfo {
        val json = Json { ignoreUnknownKeys = true }
        val jsonString = File(filename).readText()
        return json.decodeFromString(jsonString)
    }
}