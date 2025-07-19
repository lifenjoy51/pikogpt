
package data

import kotlinx.coroutines.runBlocking
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.StandardOpenOption

/**
 * `StoriesBPEPrep` 객체를 실행하여 BPE 토큰화를 수행하는 메인 함수입니다.
 * 특정 경로("data/1k")에 있는 텍스트 파일을 처리합니다.
 */
fun main() {
    StoriesBPEPrep.run("data/1k")
}

/**
 * 텍스트 데이터를 BPE(Byte Pair Encoding)를 사용하여 토큰화하고,
 * 훈련 및 검증 데이터셋으로 분할하여 저장하는 객체입니다.
 */
object StoriesBPEPrep {
    /**
     * 지정된 경로의 텍스트 파일을 읽어 BPE 토큰화를 수행하고 결과를 저장합니다.
     *
     * 1. 텍스트 파일에서 고유 단어를 추출하여 `unique_words.txt`로 저장합니다.
     * 2. `SimpleBPE`를 사용하여 텍스트 데이터로 BPE 모델을 훈련합니다.
     * 3. 훈련된 BPE 모델로 전체 텍스트를 인코딩(토큰화)합니다.
     * 4. 인코딩된 데이터를 90%는 훈련용(`train.bin`), 10%는 검증용(`val.bin`)으로 분할합니다.
     * 5. 토큰화에 사용된 어휘 사전 정보(meta)를 `meta.json` 파일로 저장합니다.
     *
     * @param path 처리할 데이터가 있는 디렉토리 경로 (e.g., "data/1k")
     */
    fun run(path: String) {
        val inputFile = File("$path/stories.txt")
        val dataDir = File(path)
        val text = inputFile.readText()

        // 고유 단어 추출 및 파일로 출력
        val uniqueWords = text.lowercase()
            .replace(Regex("[^a-z\\s]"), "")
            .split(Regex("\\s+"))
            .filter { it.isNotEmpty() }
            .toSet()
            .sorted()

        val wordsFile = File(dataDir, "unique_words.txt")
        wordsFile.writeText(uniqueWords.joinToString("\n"))
        println("Unique words saved to: ${wordsFile.absolutePath}")
        println("Total unique words: ${String.format("%,d", uniqueWords.size)}")

        // BPE 모델 훈련
        val bpe = SimpleBPE(maxVocabSize = 1000)
        runBlocking {
            bpe.train(text)
        }

        // 텍스트 인코딩
        val data = bpe.encode(text)

        // 데이터 분할 (훈련용/검증용)
        val n = data.size
        val trainData = data.subList(0, (n * 0.9).toInt())
        val valData = data.subList((n * 0.9).toInt(), n)

        println("Total tokens: ${String.format("%,d", n)}")
        println("Train tokens: ${String.format("%,d", trainData.size)}")
        println("Val tokens: ${String.format("%,d", valData.size)}")

        // 데이터 파일 저장
        writeData(trainData, File(dataDir, "train.bin"))
        writeData(valData, File(dataDir, "val.bin"))

        // 메타데이터 저장
        val vocabSize = bpe.getVocabSize()
        val itos = bpe.getItos()
        val stoi = bpe.getStoi()
        val meta = MetaInfo(vocabSize, itos, stoi)
        val json = Json { prettyPrint = true }
        File(dataDir, "meta.json").writeText(json.encodeToString(meta))

        println("Vocab size: $vocabSize")
        println("Data (first 10): ${data.take(10)}")
    }

    /**
     * 정수 리스트를 바이너리 파일로 저장합니다.
     *
     * 각 정수는 4바이트로 변환되어 파일에 기록됩니다.
     * GPT 모델이 훈련 데이터를 효율적으로 읽을 수 있는 형식입니다.
     *
     * @param data 저장할 정수 토큰 데이터 리스트
     * @param file 저장할 파일 객체
     */
    fun writeData(data: List<Int>, file: File) {
        val buffer = ByteBuffer.allocate(data.size * 4)
        for (d in data) {
            buffer.putInt(d)
        }
        buffer.flip()
        FileChannel.open(file.toPath(), StandardOpenOption.CREATE, StandardOpenOption.WRITE).use { channel ->
            channel.write(buffer)
        }
    }
}


