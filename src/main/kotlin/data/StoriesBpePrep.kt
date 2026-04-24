
package data

import kotlinx.coroutines.runBlocking
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.StandardOpenOption

/**
 * `StoriesBPEPrep`을 실행해 지정된 데이터 디렉토리의 `stories.txt`를 BPE로 토큰화한다.
 * 인자가 없으면 기본 경로 `data/simple` 을 사용.
 */
fun main(args: Array<String>) {
    val path = args.getOrNull(0) ?: "data/simple"
    StoriesBPEPrep.run(path)
}

/**
 * 텍스트 데이터를 BPE로 토큰화하고, 훈련/검증 데이터셋으로 분할 저장하는 파이프라인.
 *
 * 산출물 (`<path>/` 아래):
 *   - `train.bin` / `val.bin` : big-endian int32 토큰 시퀀스 (90:10 split)
 *   - `meta.json`             : vocab + BPE merges + 전처리 플래그 (→ 샘플러가 재생)
 *   - `unique_words.txt`      : 진단용 단어 빈도 덤프
 */
object StoriesBPEPrep {

    /** 작은 교육용 모델에 맞춘 기본 전처리. 필요 시 run() 인자로 끌 수 있다. */
    private const val DEFAULT_LOWERCASE = true
    private const val DEFAULT_WORD_PRE_TOKENIZE = true
    private const val DEFAULT_VOCAB_SIZE = 1000

    fun run(
        path: String,
        maxVocabSize: Int = DEFAULT_VOCAB_SIZE,
        lowercase: Boolean = DEFAULT_LOWERCASE,
        useWordPreTokenize: Boolean = DEFAULT_WORD_PRE_TOKENIZE,
        verbose: Boolean = true,
    ) {
        val inputFile = File("$path/stories.txt")
        val dataDir = File(path)
        val rawText = inputFile.readText()
        val normalizedText = if (lowercase) rawText.lowercase() else rawText

        // 고유 단어 빈도 덤프 (디버그용)
        val uniqueWords = normalizedText
            .replace(Regex("[^a-z\\s]"), "")
            .split(Regex("\\s+"))
            .filter { it.isNotEmpty() }
            .groupingBy { it }
            .eachCount()
            .entries.sortedByDescending { it.value }
        val wordsFile = File(dataDir, "unique_words.txt")
        wordsFile.writeText(buildString {
            uniqueWords.forEach { (word, count) -> append("$word\t$count\n") }
        })
        println("Unique words: ${String.format("%,d", uniqueWords.size)}")

        // BPE 학습 — 원본 텍스트에 대해 (SimpleBPE 내부에서 다시 lowercase 적용 가능하지만
        // 위에서 이미 한 번 했으므로 이후 flag는 false로 두고, 대신 인코딩 경로에서 일관되도록 둘 다 true)
        val bpe = SimpleBPE(
            maxVocabSize = maxVocabSize,
            lowercase = lowercase,
            useWordPreTokenize = useWordPreTokenize,
            standardBpeScoring = true,
            verbose = verbose,
        )
        runBlocking { bpe.train(rawText) }  // SimpleBPE가 lowercase 플래그에 따라 스스로 정규화

        // 텍스트 인코딩
        val encoded = bpe.encode(rawText)
        val totalCount = encoded.size
        val splitIdx = (totalCount * 0.9).toInt()
        val trainTokens = encoded.subList(0, splitIdx)
        val valTokens = encoded.subList(splitIdx, totalCount)

        println("Total tokens: ${String.format("%,d", totalCount)}")
        println("Train tokens: ${String.format("%,d", trainTokens.size)}")
        println("Val tokens:   ${String.format("%,d", valTokens.size)}")

        writeData(trainTokens, File(dataDir, "train.bin"))
        writeData(valTokens, File(dataDir, "val.bin"))

        // 메타데이터 저장 (Sampler가 정확히 같은 토큰화를 재생하려면 merges + 플래그가 필요)
        val stoi = bpe.getStoi()
        val itos = bpe.getItos()
        val mergesSerialized = bpe.getMerges().map { listOf(it.first, it.second) }
        val meta = MetaInfo(
            vocabularySize = bpe.getVocabSize(),
            indexToString = itos,
            stringToIndex = stoi,
            merges = mergesSerialized,
            lowercase = lowercase,
            useWordPreTokenize = useWordPreTokenize,
        )
        val json = Json { prettyPrint = true }
        File(dataDir, "meta.json").writeText(json.encodeToString(meta))

        println("Vocab size: ${bpe.getVocabSize()}, merges: ${mergesSerialized.size}")
        println("Saved: ${File(dataDir, "meta.json").absolutePath}")
    }

    /** 정수 토큰 리스트를 4바이트 big-endian int로 바이너리 저장. */
    fun writeData(tokens: List<Int>, file: File) {
        val buffer = ByteBuffer.allocate(tokens.size * 4)
        for (id in tokens) buffer.putInt(id)
        buffer.flip()
        FileChannel.open(
            file.toPath(),
            StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING
        ).use { channel -> channel.write(buffer) }
    }
}
