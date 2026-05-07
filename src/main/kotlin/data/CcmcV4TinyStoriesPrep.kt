package data

import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.random.Random

/**
 * CCMC v4-tinystories prep — cefr-kb의 raw.jsonl을 train.bin/val.bin으로 인코딩.
 *
 *   입력 : cefr-kb 출력 JSONL (lemma · text · tokens per line)
 *          ~/works/llm-playground/data/processed/ccmc_v4_tinystories/raw.jsonl
 *   출력 : data/ccmc-v4-tinystories/{train.bin,val.bin,meta.json}
 *          - meta.json은 ccmc-v2-pro/stage1의 BPE meta를 그대로 재사용 (vocab 2000)
 *            → stage1 ckpt에서 finetune 가능, 같은 토크나이저
 *          - .bin은 little-endian uint16 sequence (vocab ≤ 65536이라 안전)
 *
 * 사용법:
 *   ./gradlew runCcmcV4TinyStoriesPrep
 *   ./gradlew runCcmcV4TinyStoriesPrep \
 *       --args="<jsonl 경로> <out 디렉토리> <stage1 meta 경로>"
 */
fun main(args: Array<String>) {
    val rawJsonlPath = args.getOrNull(0)
        ?: System.getenv("CCMC_V4_RAW_JSONL")
        ?: "${System.getProperty("user.home")}/works/llm-playground/data/processed/ccmc_v4_tinystories/raw.jsonl"
    val outDir = args.getOrNull(1) ?: "data/ccmc-v4-tinystories"
    val stage1Meta = args.getOrNull(2) ?: "data/ccmc-v2-pro/stage1/meta.json"
    val valRatio = (args.getOrNull(3)?.toFloatOrNull() ?: 0.05f).coerceIn(0.001f, 0.5f)
    val seed = args.getOrNull(4)?.toLongOrNull() ?: 42L

    val raw = File(rawJsonlPath)
    require(raw.exists()) { "JSONL 없음: ${raw.absolutePath}" }
    val metaFile = File(stage1Meta)
    require(metaFile.exists()) { "stage1 meta 없음: ${metaFile.absolutePath}" }

    File(outDir).mkdirs()

    println("=== CCMC v4-tinystories prep ===")
    println("input  : ${raw.absolutePath}")
    println("output : $outDir")
    println("meta   : ${metaFile.absolutePath} (재사용)")
    println("val ratio: $valRatio  seed: $seed")

    // 1) JSONL 읽기 — 한 줄에 한 story. text 필드만 수집.
    val parser = Json { ignoreUnknownKeys = true }
    val texts = mutableListOf<String>()
    var skipped = 0
    raw.useLines { lines ->
        for (line in lines) {
            val trimmed = line.trim()
            if (trimmed.isEmpty() || !trimmed.startsWith("{")) continue
            val obj = try {
                parser.parseToJsonElement(trimmed).jsonObject
            } catch (e: Exception) {
                skipped++; continue
            }
            val text = obj["text"]?.jsonPrimitive?.contentOrNull
            if (text.isNullOrBlank()) { skipped++; continue }
            texts += text.trim()
        }
    }
    require(texts.isNotEmpty()) { "유효한 story 없음 (skipped=$skipped)" }
    println("stories loaded: ${texts.size} (skipped: $skipped)")

    // 2) shuffle + train/val split
    val shuffled = texts.toMutableList().also { it.shuffle(Random(seed)) }
    val nVal = (shuffled.size * valRatio).toInt().coerceAtLeast(1)
    val valStories = shuffled.take(nVal)
    val trainStories = shuffled.drop(nVal)
    println("split: train=${trainStories.size} val=${valStories.size}")

    // 3) stage1 meta + BPE 복원
    val meta = parser.decodeFromString<MetaInfo>(metaFile.readText())
    println("vocab: ${meta.vocabularySize}, BPE merges: ${meta.merges.size}")

    val encoder: (String) -> List<Int> = if (meta.merges.isNotEmpty()) {
        val bpe = SimpleBPE(
            maxVocabSize = meta.vocabularySize,
            specialTokens = meta.specialTokens,
            lowercase = meta.lowercase,
            useWordPreTokenize = meta.useWordPreTokenize,
            standardBpeScoring = true,
            verbose = false,
        )
        bpe.restore(meta.stringToIndex, meta.merges.map { it[0] to it[1] })
        ({ s -> bpe.encode(s) })
    } else {
        // merges 없는 구버전 meta — greedy fallback (성능 떨어지지만 동작은 함)
        ({ s -> greedyTokenize(s, meta.stringToIndex) })
    }

    // 4) 각 story를 인코딩하고 story 사이 BOS/EOS 같은 separator 없이 그대로 join.
    //    pikogpt의 다른 stage(예: stage1)와 동일 패턴 — 각 story가 이미
    //    1 turn (시작/끝 marker는 없음). 학습 시 record-aware sampling이 record
    //    경계 인식하려면 separator 필요할 수 있으나, 단순 시작은 single newline.
    val trainTokens = mutableListOf<Int>()
    val valTokens = mutableListOf<Int>()
    fun encodeAll(stories: List<String>, dst: MutableList<Int>) {
        for ((idx, story) in stories.withIndex()) {
            dst += encoder(story)
            // story 사이 separator — 단순 newline. record-aware는 prep 단계에서 BOS 추가하는
            // 변형도 가능 (CCMC stage1 패턴). 일단 plain.
            if (idx < stories.size - 1) {
                dst += encoder("\n")
            }
            if ((idx + 1) % 10000 == 0) {
                println("  encoded ${idx + 1}/${stories.size}")
            }
        }
    }
    println("encoding train...")
    encodeAll(trainStories, trainTokens)
    println("encoding val...")
    encodeAll(valStories, valTokens)
    println("token counts: train=${trainTokens.size}  val=${valTokens.size}")

    // 5) bin 저장 — little-endian uint16
    writeUint16Bin(File(outDir, "train.bin"), trainTokens, meta.vocabularySize)
    writeUint16Bin(File(outDir, "val.bin"), valTokens, meta.vocabularySize)

    // 6) meta.json copy
    metaFile.copyTo(File(outDir, "meta.json"), overwrite = true)

    println("=== prep done ===")
    println("  train.bin : ${trainTokens.size} tokens")
    println("  val.bin   : ${valTokens.size} tokens")
    println("  meta.json : copied from stage1")
}

private fun writeUint16Bin(file: File, tokens: List<Int>, vocabSize: Int) {
    require(vocabSize <= 65536) { "vocab too large for uint16 bin: $vocabSize" }
    file.parentFile?.mkdirs()
    RandomAccessFile(file, "rw").use { raf ->
        raf.setLength(0)
        val buf = ByteBuffer.allocate(tokens.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (t in tokens) {
            require(t in 0 until vocabSize) { "token out of vocab: $t" }
            buf.putShort(t.toShort())
        }
        raf.write(buf.array())
    }
}

private fun greedyTokenize(text: String, stoi: Map<String, Int>): List<Int> {
    val out = mutableListOf<Int>()
    val longest = stoi.keys.maxOf { it.length }
    var i = 0
    while (i < text.length) {
        var matched = false
        for (length in minOf(text.length - i, longest) downTo 1) {
            val candidate = text.substring(i, i + length)
            val id = stoi[candidate]
            if (id != null) {
                out += id
                i += length
                matched = true
                break
            }
        }
        if (!matched) {
            out += 1  // UNK
            i++
        }
    }
    return out
}
