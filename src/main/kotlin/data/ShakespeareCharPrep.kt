package data

import java.io.File
import java.net.URL
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.serialization.*
import kotlinx.serialization.json.*


class ShakespeareCharPrep {
    private val inputFilePath = "input.txt"
    private val dataUrl = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    fun prepare() {
        // 데이터셋 다운로드
        val data = downloadOrLoadData()
        println("데이터셋 문자 수: ${"%,d".format(data.length)}")
        
        // 고유 문자 추출
        val chars = data.toSet().sorted()
        val vocabSize = chars.size
        println("고유 문자들: ${chars.joinToString("")}")
        println("어휘 크기: ${"%,d".format(vocabSize)}")
        
        // 문자-정수 매핑 생성
        val stoi = chars.withIndex().associate { (i, ch) -> ch.toString() to i }
        val itos = chars.withIndex().associate { (i, ch) -> i to ch.toString() }
        
        // 인코더/디코더 함수
        fun encode(s: String): List<Int> = s.map { stoi[it.toString()]!! }
        fun decode(l: List<Int>): String = l.joinToString("") { itos[it]!! }
        
        // 훈련/검증 데이터 분할
        val n = data.length
        val trainData = data.substring(0, (n * 0.9).toInt())
        val valData = data.substring((n * 0.9).toInt())
        
        // 정수로 인코딩
        val trainIds = encode(trainData)
        val valIds = encode(valData)
        println("훈련 토큰 수: ${"%,d".format(trainIds.size)}")
        println("검증 토큰 수: ${"%,d".format(valIds.size)}")
        
        // 바이너리 파일로 저장
        saveAsBinary("train.bin", trainIds)
        saveAsBinary("val.bin", valIds)
        
        // 메타 정보 저장
        val meta = MetaInfo(
            vocabSize = vocabSize,
            itos = itos,
            stoi = stoi
        )
        saveMetaInfo("meta.json", meta)
        
        println("\n데이터 준비 완료!")
        println("생성된 파일:")
        println("- train.bin: 훈련 데이터")
        println("- val.bin: 검증 데이터")
        println("- meta.json: 메타 정보")
    }
    
    private fun downloadOrLoadData(): String {
        val file = File(inputFilePath)
        
        return if (!file.exists()) {
            println("데이터셋 다운로드 중...")
            val content = URL(dataUrl).readText()
            file.writeText(content)
            println("다운로드 완료!")
            content
        } else {
            println("기존 데이터셋 로드 중...")
            file.readText()
        }
    }
    
    private fun saveAsBinary(filename: String, ids: List<Int>) {
        val file = File(filename)
        val buffer = ByteBuffer.allocate(ids.size * 2) // uint16 = 2 bytes
        buffer.order(ByteOrder.LITTLE_ENDIAN)
        
        ids.forEach { id ->
            buffer.putShort(id.toShort())
        }
        
        file.writeBytes(buffer.array())
    }
    
    private fun saveMetaInfo(filename: String, meta: MetaInfo) {
        val json = Json { 
            prettyPrint = true 
            encodeDefaults = true
        }
        val jsonString = json.encodeToString(meta)
        File(filename).writeText(jsonString)
    }
}
