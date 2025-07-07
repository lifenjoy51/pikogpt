package data

import java.io.File
import java.net.URL
import java.nio.ByteBuffer
import java.nio.ByteOrder

class ShakespeareBPEPrep {
    private val inputFilePath = "input.txt"
    private val dataUrl = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    fun prepare() {
        // 데이터셋 다운로드 또는 로드
        val data = downloadOrLoadData()
        println("데이터셋 크기: ${"%,d".format(data.length)} 문자")
        
        // 훈련/검증 데이터 분할
        val n = data.length
        val trainData = data.substring(0, (n * 0.9).toInt())
        val valData = data.substring((n * 0.9).toInt())
        
        println("훈련 데이터: ${"%,d".format(trainData.length)} 문자")
        println("검증 데이터: ${"%,d".format(valData.length)} 문자")
        
        // BPE 학습
        println("\n=== BPE 학습 ===")
        val bpe = SimpleBPE(numMerges = 2000) // 2000번 병합
        bpe.train(trainData) // 훈련 데이터로 BPE 학습
        
        // 인코딩
        println("\n=== 데이터 인코딩 ===")
        println("훈련 데이터 인코딩 중...")
        val trainIds = bpe.encode(trainData)
        println("검증 데이터 인코딩 중...")
        val valIds = bpe.encode(valData)
        
        println("\n인코딩 결과:")
        println("훈련 토큰 수: ${"%,d".format(trainIds.size)}")
        println("검증 토큰 수: ${"%,d".format(valIds.size)}")
        println("압축률: ${String.format("%.2f", trainData.length.toDouble() / trainIds.size)}:1")
        
        // 바이너리 파일로 저장
        saveAsBinary("train.bin", trainIds)
        saveAsBinary("val.bin", valIds)
        
        // 메타 정보 저장
        saveMetaInfo("bpe_meta.txt", bpe)
        
        println("\n완료! 생성된 파일:")
        println("- train.bin (${File("train.bin").length()} bytes)")
        println("- val.bin (${File("val.bin").length()} bytes)")
        println("- bpe_meta.txt (BPE 정보)")
        
        // 샘플 출력
        println("\n=== 인코딩 샘플 ===")
        val sampleText = trainData.substring(0, 100)
        val sampleEncoded = bpe.encode(sampleText)
        println("원본 텍스트 (100자):")
        println(sampleText)
        println("\n인코딩된 토큰 (${sampleEncoded.size}개):")
        println(sampleEncoded.take(50).joinToString(" "))
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
        val buffer = ByteBuffer.allocate(ids.size * 2)
        buffer.order(ByteOrder.LITTLE_ENDIAN)
        
        ids.forEach { id ->
            buffer.putShort(id.toShort())
        }
        
        file.writeBytes(buffer.array())
    }
    
    private fun saveMetaInfo(filename: String, bpe: SimpleBPE) {
        val file = File(filename)
        val content = buildString {
            appendLine("SimpleBPE 메타 정보")
            appendLine("===================")
            appendLine("어휘 크기: ${bpe.getVocabSize()}")
            appendLine("병합 규칙 수: ${bpe.numMerges}")
            appendLine("생성 시간: ${java.time.LocalDateTime.now()}")
        }
        file.writeText(content)
    }
}
