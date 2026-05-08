
package data

import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.StandardOpenOption

/**
 * `BpePrep`을 실행해 지정된 데이터 디렉토리의 `train.txt` / `val.txt`로 BPE를 학습·인코딩한다.
 * 인자: args[0] = path (기본 `data/simple`), args[1] = vocab size (기본 1000),
 *       args[2..] flags = "skip-bin"(meta.json만), "cased"(대소문자 보존).
 *
 * 사용 예:
 *   ./gradlew runBpe --args="data/two-stage-v3/shared 2000"
 *   ./gradlew runBpe --args="data/two-stage-v3/shared 2000 cased"
 *   ./gradlew runBpe --args="data/two-stage-v3/shared 2000 skip-bin"  # OOM 회피
 */
fun main(args: Array<String>) {
    val path = args.getOrNull(0) ?: "data/simple"
    val vocab = args.getOrNull(1)?.toIntOrNull()
    val skipBin = args.any { it.equals("skip-bin", ignoreCase = true) }
    // cased: 대소문자 보존(lowercase=false). 기본은 lowercase=true 유지.
    val cased = args.any { it.equals("cased", ignoreCase = true) }
    val lowercase = !cased
    if (vocab != null) {
        BpePrep.run(path, maxVocabSize = vocab, lowercase = lowercase, skipBinOutput = skipBin)
    } else {
        BpePrep.run(path, lowercase = lowercase, skipBinOutput = skipBin)
    }
}

/**
 * 텍스트 데이터를 BPE로 토큰화하고, 훈련/검증 데이터셋으로 저장하는 파이프라인.
 *
 * 입력 규약: `<path>/train.txt` + `<path>/val.txt` 둘 다 필요.
 *   - BPE는 train.txt에서만 학습 (val leakage 방지)
 *   - val.txt는 encode 대상으로만 사용
 *   - train/val.bin은 각 파일의 토큰 전량
 *
 * 산출물 (`<path>/` 아래):
 *   - `train.bin` / `val.bin` : big-endian int32 토큰 시퀀스
 *   - `meta.json`             : vocab + BPE merges + 전처리 플래그 (→ 샘플러가 재생)
 *   - `unique_words.txt`      : 진단용 단어 빈도 덤프 (train 소스 기준)
 */
object BpePrep {

    /** 작은 교육용 모델에 맞춘 기본 전처리. 필요 시 run() 인자로 끌 수 있다. */
    private const val DEFAULT_LOWERCASE = true
    private const val DEFAULT_WORD_PRE_TOKENIZE = true
    private const val DEFAULT_VOCAB_SIZE = 1000

    /** unique_words.txt 분석 시 special token이 인접 단어와 합쳐지지 않도록 격리할 토큰들. */
    private val SPECIAL_TOKENS_FOR_ANALYSIS = listOf("<|eos|>", "<|unk|>", "<|bos|>", "<|turn|>", "<|sep|>")

    fun run(
        path: String,
        maxVocabSize: Int = DEFAULT_VOCAB_SIZE,
        lowercase: Boolean = DEFAULT_LOWERCASE,
        useWordPreTokenize: Boolean = DEFAULT_WORD_PRE_TOKENIZE,
        verbose: Boolean = true,
        /** true면 train.bin/val.bin은 만들지 않고 meta.json만 작성. 큰 코퍼스에서 encode 단계 OOM 회피용. */
        skipBinOutput: Boolean = false,
    ) {
        val dataDir = File(path)
        val trainFile = File(path, "train.txt")
        val valFile = File(path, "val.txt")

        require(trainFile.exists() && valFile.exists()) {
            "입력 파일 없음: $path 에 train.txt 와 val.txt 둘 다 필요"
        }
        println("입력: train.txt + val.txt → vocab은 train에서만 학습")

        val trainText = trainFile.readText()
        val valText = valFile.readText()

        // 고유 단어 빈도 덤프 (디버그용) — train 소스 기준.
        // special token(`<|turn|>` 등)이 인접 발화와 공백 없이 붙어있을 때
        // 정규식 [^a-z\s] 단순 제거가 `okay<|turn|>okay` → `okayturnokay`로 합쳐버리는
        // 문제를 막기 위해, 미리 special token 주변에 공백을 삽입한 뒤 정규화한다.
        val analysisText = run {
            var t = if (lowercase) trainText.lowercase() else trainText
            for (token in SPECIAL_TOKENS_FOR_ANALYSIS) {
                t = t.replace(token.lowercase(), " $token ")
            }
            t
        }
        val uniqueWords = analysisText
            // 알파벳/공백 외 chars를 *공백*으로 치환 (제거 시 `word1.word2`가 `word1word2`로 합쳐짐).
            .replace(Regex("[^a-z\\s]"), " ")
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
        val bpe = CharBPE(
            maxVocabSize = maxVocabSize,
            lowercase = lowercase,
            useWordPreTokenize = useWordPreTokenize,
            standardBpeScoring = true,
            verbose = verbose,
        )
        bpe.train(trainText)

        // 메타데이터 즉시 저장 — encode 단계가 OOM/SIGTERM으로 죽어도 BPE 학습 결과는 보존.
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
        println("Saved meta: ${File(dataDir, "meta.json").absolutePath}")

        if (skipBinOutput) {
            println("skipBinOutput=true → train.bin/val.bin 건너뜀. 학습용 인코딩은 별도 task에서.")
            return
        }

        val trainTokens = bpe.encode(trainText)
        val valTokens = bpe.encode(valText)

        println("Train tokens: ${String.format("%,d", trainTokens.size)}")
        println("Val tokens:   ${String.format("%,d", valTokens.size)}")

        writeData(trainTokens, File(dataDir, "train.bin"))
        writeData(valTokens, File(dataDir, "val.bin"))
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
