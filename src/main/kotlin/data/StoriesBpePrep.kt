
package data

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
 * 텍스트 데이터를 BPE로 토큰화하고, 훈련/검증 데이터셋으로 저장하는 파이프라인.
 *
 * 입력 규약 (우선순위):
 *   1. `train.txt` + `val.txt`가 둘 다 있으면 → **분리 입력 경로**
 *      - BPE는 train.txt에서만 학습 (val leakage 방지), val은 encode만
 *      - train/val.bin은 각 파일의 토큰 전량
 *   2. 둘 중 하나라도 없으면 → **fallback: `stories.txt` 90:10 cut**
 *      - 단일 파일에서 BPE 학습 + encode 후 앞 90% / 뒤 10% 순차 절단
 *
 * 산출물 (`<path>/` 아래):
 *   - `train.bin` / `val.bin` : big-endian int32 토큰 시퀀스
 *   - `meta.json`             : vocab + BPE merges + 전처리 플래그 (→ 샘플러가 재생)
 *   - `unique_words.txt`      : 진단용 단어 빈도 덤프 (train 소스 기준)
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
        val dataDir = File(path)
        val trainFile = File(path, "train.txt")
        val valFile = File(path, "val.txt")
        val storiesFile = File(path, "stories.txt")

        val split: SplitSource = when {
            trainFile.exists() && valFile.exists() -> {
                println("분리 입력 감지: train.txt + val.txt → vocab은 train에서만 학습")
                SplitSource.Separate(trainFile.readText(), valFile.readText())
            }
            storiesFile.exists() -> {
                println("단일 입력: stories.txt → 토큰 90:10 순차 분할 (fallback)")
                SplitSource.Combined(storiesFile.readText())
            }
            else -> error("입력 파일 없음: $path 에 {train.txt+val.txt} 또는 stories.txt 중 하나 필요")
        }

        val trainText = when (split) {
            is SplitSource.Separate -> split.trainText
            is SplitSource.Combined -> split.text
        }

        // 고유 단어 빈도 덤프 (디버그용) — train 소스 기준. 정규화는 이 한 번만.
        val analysisText = if (lowercase) trainText.lowercase() else trainText
        val uniqueWords = analysisText
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

        // BPE 학습 — train 소스에서만. val 텍스트는 encode 대상으로만 사용해 leakage 방지.
        val bpe = SimpleBPE(
            maxVocabSize = maxVocabSize,
            lowercase = lowercase,
            useWordPreTokenize = useWordPreTokenize,
            standardBpeScoring = true,
            verbose = verbose,
        )
        bpe.train(trainText)

        val (trainTokens, valTokens) = when (split) {
            is SplitSource.Separate -> {
                val tr = bpe.encode(split.trainText)
                val vl = bpe.encode(split.valText)
                tr to vl
            }
            is SplitSource.Combined -> {
                val encoded = bpe.encode(split.text)
                val splitIdx = (encoded.size * 0.9).toInt()
                encoded.subList(0, splitIdx) to encoded.subList(splitIdx, encoded.size)
            }
        }

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

    /**
     * 분리/결합 입력 경로를 표현하는 sealed 소스 타입. `run()` 내부 분기 용.
     */
    private sealed class SplitSource {
        class Separate(val trainText: String, val valText: String) : SplitSource()
        class Combined(val text: String) : SplitSource()
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
