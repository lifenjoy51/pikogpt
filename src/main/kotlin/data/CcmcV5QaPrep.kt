package data

import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.random.Random

/**
 * v5_qa Stage 2 산출 dialogues → 학습용 token bin 변환.
 *
 * 입력:
 *   ~/works/llm-playground/data/processed/ccmc_v5_qa_dialogues/flash/raw.jsonl
 *   - 한 줄당 1 dialogue. ok=true인 record만 사용 (~14,699건).
 *   - dialogue.turns: ["Q1", "A1", "Q2", "A2", ...] (짝수 인덱스 Q, 홀수 A).
 *
 * 처리:
 *   1) ok=true & dialogue 비어있지 않은 record만 추출
 *   2) text = "<|bos|>turn1<|turn|>turn2<|turn|>...<|eos|>"
 *   3) text MD5 dedupe
 *   4) record-level shuffle + 95:5 split
 *   5) ccmc-v2-pro/stage1/meta.json BPE meta 재사용 (vocab=2000, special token id 동일)
 *   6) <|bos|>/<|turn|>/<|eos|>는 stringToIndex 직접 lookup으로 single-id 보장
 *      (useWordPreTokenize=true에서 encode("<|bos|>")가 [space, bos]로 split될 수 있음)
 *
 * 출력: data/ccmc-v5-qa/{train.bin, val.bin, meta.json, prep_manifest.json}
 *
 * 사용법:
 *   ./gradlew runCcmcV5QaPrep
 *   ./gradlew runCcmcV5QaPrep --args="<rawJsonl> <outDir> <metaPath> <valRatio> <seed>"
 */
fun main(args: Array<String>) {
    val rawJsonl = args.getOrNull(0)
        ?: "${System.getProperty("user.home")}/works/llm-playground/data/processed/ccmc_v5_qa_dialogues/flash/raw.jsonl"
    val outDir = args.getOrNull(1) ?: "data/ccmc-v5-qa"
    val metaPath = args.getOrNull(2) ?: "data/ccmc-v2-pro/stage1/meta.json"
    val valRatio = (args.getOrNull(3)?.toFloatOrNull() ?: 0.05f).coerceIn(0.001f, 0.5f)
    val seed = args.getOrNull(4)?.toLongOrNull() ?: 42L

    println("=== CCMC v5-qa prep ===")
    println("raw      : $rawJsonl")
    println("out dir  : $outDir")
    println("meta     : $metaPath")
    println("val ratio: $valRatio  seed: $seed")

    val parser = Json { ignoreUnknownKeys = true }
    val metaFile = File(metaPath)
    require(metaFile.exists()) { "meta 없음: ${metaFile.absolutePath}" }
    val meta = parser.decodeFromString<MetaInfo>(metaFile.readText())
    println("vocab=${meta.vocabularySize}, merges=${meta.merges.size}")

    val bpe = CharBPE(
        maxVocabSize = meta.vocabularySize,
        specialTokens = meta.specialTokens,
        lowercase = meta.lowercase,
        useWordPreTokenize = meta.useWordPreTokenize,
        standardBpeScoring = true,
        verbose = false,
    )
    bpe.restore(meta.stringToIndex, meta.merges.map { it[0] to it[1] })
    val bosId = meta.stringToIndex["<|bos|>"] ?: error("<|bos|> not in meta")
    val eosId = meta.stringToIndex["<|eos|>"] ?: error("<|eos|> not in meta")
    val turnId = meta.stringToIndex["<|turn|>"] ?: error("<|turn|> not in meta")
    println("BPE restored. <|bos|>=$bosId  <|turn|>=$turnId  <|eos|>=$eosId (직접 id 사용)")

    // ── load dialogues ─────────────────────────────────────────────────────────
    val rawFile = File(rawJsonl)
    require(rawFile.exists()) { "raw.jsonl 없음: ${rawFile.absolutePath}" }
    var totalLines = 0
    var skippedFail = 0
    var skippedEmpty = 0
    val rawDialogueRecords = mutableListOf<DialogueRecord>()
    rawFile.useLines { lines ->
        for (line in lines) {
            val t = line.trim()
            if (t.isEmpty() || !t.startsWith("{")) continue
            totalLines++
            val obj = try { parser.parseToJsonElement(t).jsonObject } catch (_: Exception) { continue }
            val ok = obj["ok"]?.jsonPrimitive?.booleanOrNull == true
            if (!ok) { skippedFail++; continue }
            val turnsArr = obj["dialogue"]?.jsonObject?.get("turns")?.jsonArray
            if (turnsArr == null) { skippedEmpty++; continue }
            val turns = turnsArr
                .mapNotNull { it.jsonPrimitive.contentOrNull?.trim() }
                .filter { it.isNotEmpty() }
            if (turns.isEmpty() || turns.size % 2 != 0) { skippedEmpty++; continue }
            // body: turn1 + <|turn|> + turn2 + ... (BOS/EOS는 인코딩 단계에서 직접 id로 prepend/append)
            val body = turns.joinToString(separator = "<|turn|>")
            rawDialogueRecords += DialogueRecord(md5(body), turns)
        }
    }
    println("loaded: total=$totalLines  ok_used=${rawDialogueRecords.size}  skipped_fail=$skippedFail  skipped_empty=$skippedEmpty")

    // ── dedupe (text MD5) ──────────────────────────────────────────────────────
    val seenKeys = HashSet<String>()
    val unique = rawDialogueRecords.filter { seenKeys.add(it.key) }
    val dedupCount = rawDialogueRecords.size - unique.size
    println("dedupe: ${rawDialogueRecords.size} → ${unique.size} (-$dedupCount)")

    // ── shuffle + split ────────────────────────────────────────────────────────
    val shuffled = unique.toMutableList().apply { shuffle(Random(seed)) }
    val nVal = (shuffled.size * valRatio).toInt().coerceAtLeast(1)
    val valRec = shuffled.take(nVal)
    val trainRec = shuffled.drop(nVal)
    println("split: train=${trainRec.size} val=${valRec.size}")

    val valKeys = valRec.mapTo(HashSet(valRec.size)) { it.key }
    val trainKeys = trainRec.mapTo(HashSet(trainRec.size)) { it.key }
    require(valKeys.intersect(trainKeys).isEmpty()) { "val/train leak detected" }

    // ── encode ─────────────────────────────────────────────────────────────────
    // turn 단위로 인코딩하고 그 사이에 turnId/bosId/eosId 직접 삽입.
    // 단일 body 통째 인코딩하면 useWordPreTokenize에서 <|turn|>가 다중 토큰화될 수 있음.
    fun encodeDialogueRecord(r: DialogueRecord): List<Int> {
        val out = ArrayList<Int>(r.turns.size * 16 + 2)
        out.add(bosId)
        for ((i, turn) in r.turns.withIndex()) {
            if (i > 0) out.add(turnId)
            out.addAll(bpe.encode(turn))
        }
        out.add(eosId)
        return out
    }
    fun encodeAll(label: String, recs: List<DialogueRecord>): List<Int> {
        val out = ArrayList<Int>(recs.size * 200)
        for ((idx, r) in recs.withIndex()) {
            out.addAll(encodeDialogueRecord(r))
            if ((idx + 1) % 2000 == 0) println("  [$label] encoded ${idx + 1}/${recs.size}")
        }
        return out
    }
    println("encoding train...")
    val trainTokens = encodeAll("train", trainRec)
    println("encoding val...")
    val valTokens = encodeAll("val", valRec)
    println("token totals: train=${trainTokens.size}  val=${valTokens.size}")

    // ── output ─────────────────────────────────────────────────────────────────
    File(outDir).mkdirs()
    writeUint16Bin(File(outDir, "train.bin"), trainTokens, meta.vocabularySize)
    writeUint16Bin(File(outDir, "val.bin"), valTokens, meta.vocabularySize)
    metaFile.copyTo(File(outDir, "meta.json"), overwrite = true)
    writeManifest(
        File(outDir, "prep_manifest.json"),
        rawJsonl, totalLines, rawDialogueRecords.size, dedupCount,
        trainRec.size, valRec.size, trainTokens.size, valTokens.size,
        seed, valRatio,
    )

    println("=== prep done ===")
    println("  train.bin : ${trainTokens.size} tokens  (${trainRec.size} dialogues)")
    println("  val.bin   : ${valTokens.size} tokens  (${valRec.size} dialogues)")
    println("  meta.json : copied from ${metaFile.name}")
    println("  manifest  : ${File(outDir, "prep_manifest.json").path}")
}

// ── data class ─────────────────────────────────────────────────────────────────
private data class DialogueRecord(
    val key: String,         // body MD5 (dedupe key)
    val turns: List<String>,
)

// ── output writers ─────────────────────────────────────────────────────────────
private fun writeUint16Bin(file: File, tokens: List<Int>, vocabSize: Int) {
    // 명칭은 uint16이지만 vec/scalar DataLoader가 BIG_ENDIAN int32로 읽으므로 4 bytes BE int32로 저장.
    file.parentFile?.mkdirs()
    RandomAccessFile(file, "rw").use { raf ->
        raf.setLength(0)
        val buf = ByteBuffer.allocate(tokens.size * 4).order(ByteOrder.BIG_ENDIAN)
        for (t in tokens) {
            require(t in 0 until vocabSize) { "token out of vocab: $t" }
            buf.putInt(t)
        }
        raf.write(buf.array())
    }
}

private fun writeManifest(
    file: File,
    rawJsonl: String,
    totalLines: Int,
    okDialogueRecords: Int,
    dedupCount: Int,
    trainDialogues: Int,
    valDialogues: Int,
    trainTokens: Int,
    valTokens: Int,
    seed: Long,
    valRatio: Float,
) {
    val sb = StringBuilder()
    sb.append("{\n")
    sb.append("  \"source_jsonl\": \"${rawJsonl.replace("\\", "\\\\")}\",\n")
    sb.append("  \"total_lines\": $totalLines,\n")
    sb.append("  \"ok_records\": $okDialogueRecords,\n")
    sb.append("  \"deduped_records\": $dedupCount,\n")
    sb.append("  \"train_dialogues\": $trainDialogues,\n")
    sb.append("  \"val_dialogues\": $valDialogues,\n")
    sb.append("  \"train_tokens\": $trainTokens,\n")
    sb.append("  \"val_tokens\": $valTokens,\n")
    sb.append("  \"seed\": $seed,\n")
    sb.append("  \"val_ratio\": $valRatio\n")
    sb.append("}\n")
    file.writeText(sb.toString())
}

private fun md5(s: String): String {
    val bytes = MessageDigest.getInstance("MD5").digest(s.toByteArray(Charsets.UTF_8))
    return buildString(bytes.size * 2) { for (b in bytes) append("%02x".format(b)) }
}
