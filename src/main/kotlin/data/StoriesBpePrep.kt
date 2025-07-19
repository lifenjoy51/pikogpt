
package data

import kotlinx.coroutines.runBlocking
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.StandardOpenOption

fun main() {
    StoriesBPEPrep.run("data/1k")
}

object StoriesBPEPrep {
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

        val bpe = SimpleBPE(maxVocabSize = 1000)
        runBlocking {
            bpe.train(text)
        }

        val data = bpe.encode(text)

        val n = data.size
        val trainData = data.subList(0, (n * 0.9).toInt())
        val valData = data.subList((n * 0.9).toInt(), n)

        println("Total tokens: ${String.format("%,d", n)}")
        println("Train tokens: ${String.format("%,d", trainData.size)}")
        println("Val tokens: ${String.format("%,d", valData.size)}")

        writeData(trainData, File(dataDir, "train.bin"))
        writeData(valData, File(dataDir, "val.bin"))

        val vocabSize = bpe.getVocabSize()
        val itos = bpe.getItos()
        val stoi = bpe.getStoi()
        val meta = MetaInfo(vocabSize, itos, stoi)
        val json = Json { prettyPrint = true }
        File(dataDir, "meta.json").writeText(json.encodeToString(meta))

        println("Vocab size: $vocabSize")
        println("Data: ${data.take(10)}")
    }

    fun writeData(data: List<Int>, file: File) {
        val buffer = ByteBuffer.allocate(data.size * 4)
        for (d in data) {
            buffer.putInt(d)
        }
        buffer.flip()
        val channel = FileChannel.open(file.toPath(), StandardOpenOption.CREATE, StandardOpenOption.WRITE)
        channel.write(buffer)
        channel.close()
    }
}


