package data

import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.random.Random

/**
 * v2-pro + v4 9 epoch 통합 prep.
 *
 * 입력:
 *   v2-pro: data/ccmc-v2-pro/{stage1,stage2}/{train,val}.txt (4 파일, 단일 source "v2_pro")
 *   v4:     ~/works/llm-playground/data/processed/ccmc_v4_tinystories/{flash,flash_e2..e6,pro,pro_e2,pro_e3}/raw.jsonl
 *
 * 처리:
 *   - v4 ban-word filter (arthopods/adios/amor/asante/gracias 포함 stories 제외)
 *   - text MD5 dedupe (v2-pro 우선, v4 폴더 간 중복 제거)
 *   - record-level shuffle + 95:5 split
 *   - v2-pro: 그대로 인코딩 (stage1/2 marker 보존)
 *   - v4: BOS + text + EOS로 wrap
 *   - stage1 BPE meta 재사용 (vocab=2000)
 *
 * 출력: data/ccmc-v4-merged/{train.bin, val.bin, meta.json, prep_manifest.json}
 *
 * 사용법:
 *   ./gradlew runCcmcV4MergedPrep
 *   ./gradlew runCcmcV4MergedPrep --args="<pikoData> <v4Base> <outDir> <metaPath> <valRatio> <seed>"
 */
fun main(args: Array<String>) {
    val pikoData = args.getOrNull(0) ?: "data"
    val v4Base = args.getOrNull(1)
        ?: "${System.getProperty("user.home")}/works/llm-playground/data/processed/ccmc_v4_tinystories"
    val outDir = args.getOrNull(2) ?: "data/ccmc-v4-merged"
    val stage1Meta = args.getOrNull(3) ?: "$pikoData/ccmc-v2-pro/stage1/meta.json"
    val valRatio = (args.getOrNull(4)?.toFloatOrNull() ?: 0.05f).coerceIn(0.001f, 0.5f)
    val seed = args.getOrNull(5)?.toLongOrNull() ?: 42L

    println("=== CCMC v4-merged prep ===")
    println("piko data : $pikoData")
    println("v4 base   : $v4Base")
    println("out dir   : $outDir")
    println("meta      : $stage1Meta")
    println("val ratio : $valRatio  seed: $seed")

    val parser = Json { ignoreUnknownKeys = true }
    val metaFile = File(stage1Meta)
    require(metaFile.exists()) { "stage1 meta 없음: ${metaFile.absolutePath}" }
    val meta = parser.decodeFromString<MetaInfo>(metaFile.readText())
    println("vocab=${meta.vocabularySize}, merges=${meta.merges.size}, specialTokens=${meta.specialTokens}")

    // ── BPE 복원 + special token 단일 토큰 검증 ─────────────────────────────────
    val bpe = CharBPE(
        maxVocabSize = meta.vocabularySize,
        specialTokens = meta.specialTokens,
        lowercase = meta.lowercase,
        useWordPreTokenize = meta.useWordPreTokenize,
        standardBpeScoring = true,
        verbose = false,
    )
    bpe.restore(meta.stringToIndex, meta.merges.map { it[0] to it[1] })
    val bosId = meta.stringToIndex["<|bos|>"] ?: error("BOS not in meta")
    val eosId = meta.stringToIndex["<|eos|>"] ?: error("EOS not in meta")
    // useWordPreTokenize=true면 <|bos|> 텍스트 인코딩 시 space prefix가 붙어
    // [space, bos] = [6, 2] 식으로 multi-token이 될 수 있음. v4 wrapping에서는
    // 이를 우회하려 stringToIndex 직접 lookup한 single id를 사용.
    println("BPE restored. <|bos|>=$bosId  <|eos|>=$eosId (직접 id 사용)")
    println("  encode('<|bos|>') = ${bpe.encode("<|bos|>")}")
    println("  encode('<|eos|>') = ${bpe.encode("<|eos|>")}")

    // ── v2-pro records (단일 source "v2_pro") ───────────────────────────────────
    val records = mutableListOf<Record>()
    val v2Paths = listOf(
        "$pikoData/ccmc-v2-pro/stage1/train.txt",
        "$pikoData/ccmc-v2-pro/stage1/val.txt",
        "$pikoData/ccmc-v2-pro/stage2/train.txt",
        "$pikoData/ccmc-v2-pro/stage2/val.txt",
    )
    var v2Lines = 0
    for (path in v2Paths) {
        val before = records.size
        records += loadV2Lines(File(path), source = "v2_pro", needsBosEos = false)
        val added = records.size - before
        v2Lines += added
        println("  v2-pro [$path]: $added records")
    }
    println("v2-pro total: $v2Lines records")

    // ── v4 records (폴더별 source, ban-word filter) ─────────────────────────────
    val v4Folders = listOf(
        "flash", "flash_e2", "flash_e3", "flash_e4", "flash_e5", "flash_e6",
        "pro", "pro_e2", "pro_e3",
    )
    val banWords = setOf("arthopods", "adios", "amor", "asante", "gracias")
    val filteredCounts = mutableMapOf<String, Int>()
    var v4Total = 0
    for (folder in v4Folders) {
        val path = File("$v4Base/$folder/raw.jsonl")
        if (!path.exists()) {
            println("  [skip] v4 폴더 없음: $path")
            continue
        }
        val before = records.size
        val (loaded, filtered) = loadV4Jsonl(
            path, source = "v4_$folder", needsBosEos = true, banWords = banWords, parser = parser,
        )
        records += loaded
        filteredCounts.merge(folder, filtered) { a, b -> a + b }
        val added = records.size - before
        v4Total += added
        println("  v4 [$folder]: $added records (filtered=$filtered)")
    }
    val totalFiltered = filteredCounts.values.sum()
    println("v4 total: $v4Total records (filtered out: $totalFiltered)")

    // ── text MD5 dedupe (v2-pro 우선) ──────────────────────────────────────────
    val seenKeys = HashSet<String>()
    val unique = records.filter { seenKeys.add(it.key) }
    val dedupCount = records.size - unique.size
    println("dedupe: ${records.size} → ${unique.size} (-$dedupCount)")

    // ── shuffle + 95:5 split ────────────────────────────────────────────────────
    val shuffled = unique.toMutableList().apply { shuffle(Random(seed)) }
    val nVal = (shuffled.size * valRatio).toInt().coerceAtLeast(1)
    val valRec = shuffled.take(nVal)
    val trainRec = shuffled.drop(nVal)
    println("split: train=${trainRec.size} val=${valRec.size}")

    // val/train leak assert
    val valKeys = valRec.mapTo(HashSet(valRec.size)) { it.key }
    val trainKeys = trainRec.mapTo(HashSet(trainRec.size)) { it.key }
    require(valKeys.intersect(trainKeys).isEmpty()) { "val/train leak detected" }

    // ── encode ─────────────────────────────────────────────────────────────────
    val sourceTokenCounts = LinkedHashMap<String, Long>()
    fun encodeRecord(r: Record): List<Int> {
        val core = bpe.encode(r.text)
        return if (r.needsBosEos) {
            val out = ArrayList<Int>(core.size + 2)
            out.add(bosId); out.addAll(core); out.add(eosId)
            out
        } else core
    }
    fun encodeAll(recs: List<Record>): List<Int> {
        val out = ArrayList<Int>(recs.size * 200)
        for ((idx, r) in recs.withIndex()) {
            val toks = encodeRecord(r)
            out.addAll(toks)
            sourceTokenCounts.merge(r.source, toks.size.toLong()) { a, b -> a + b }
            if ((idx + 1) % 5000 == 0) println("  encoded ${idx + 1}/${recs.size}")
        }
        return out
    }
    println("encoding train...")
    val trainTokens = encodeAll(trainRec)
    println("encoding val...")
    val valTokens = encodeAll(valRec)
    println("token totals: train=${trainTokens.size}  val=${valTokens.size}")

    // ── output ─────────────────────────────────────────────────────────────────
    File(outDir).mkdirs()
    writeUint16Bin(File(outDir, "train.bin"), trainTokens, meta.vocabularySize)
    writeUint16Bin(File(outDir, "val.bin"), valTokens, meta.vocabularySize)
    metaFile.copyTo(File(outDir, "meta.json"), overwrite = true)
    writeManifest(
        File(outDir, "prep_manifest.json"),
        sourceTokenCounts, trainTokens.size, valTokens.size,
        seed, valRatio, totalFiltered, dedupCount,
    )

    println("=== prep done ===")
    println("  train.bin : ${trainTokens.size} tokens")
    println("  val.bin   : ${valTokens.size} tokens")
    println("  meta.json : copied from ${metaFile.name}")
    println("  source breakdown:")
    sourceTokenCounts.forEach { (s, c) -> println("    $s : $c tokens") }
    println("  filtered out (v4 ban-word): $totalFiltered")
    println("  deduped: $dedupCount")
}

// ── data class ─────────────────────────────────────────────────────────────────
private data class Record(
    val source: String,
    val key: String,           // text MD5 (dedupe key)
    val text: String,
    val needsBosEos: Boolean,
)

// ── loaders ────────────────────────────────────────────────────────────────────
private fun loadV2Lines(file: File, source: String, needsBosEos: Boolean): List<Record> {
    if (!file.exists()) return emptyList()
    return file.useLines { lines ->
        lines.mapNotNull { l ->
            val t = l.trim()
            if (t.isEmpty()) null else Record(source, md5(t), t, needsBosEos)
        }.toList()
    }
}

private fun loadV4Jsonl(
    file: File,
    source: String,
    needsBosEos: Boolean,
    banWords: Set<String>,
    parser: Json,
): Pair<List<Record>, Int> {
    val out = mutableListOf<Record>()
    var filtered = 0
    file.useLines { lines ->
        for (line in lines) {
            val t = line.trim()
            if (t.isEmpty() || !t.startsWith("{")) continue
            val obj = try { parser.parseToJsonElement(t).jsonObject } catch (_: Exception) { continue }
            val text = obj["text"]?.jsonPrimitive?.contentOrNull?.trim() ?: continue
            if (text.isEmpty()) continue
            if (containsBanned(text, banWords)) {
                filtered++
                continue
            }
            out += Record(source, md5(text), text, needsBosEos)
        }
    }
    return out to filtered
}

private val WORD_RE = Regex("[a-z][a-z'-]*")

private fun containsBanned(text: String, banWords: Set<String>): Boolean {
    if (banWords.isEmpty()) return false
    val lower = text.lowercase()
    for (m in WORD_RE.findAll(lower)) {
        if (m.value in banWords) return true
    }
    return false
}

// ── output writers ─────────────────────────────────────────────────────────────
private fun writeUint16Bin(file: File, tokens: List<Int>, vocabSize: Int) {
    // 명칭은 uint16이지만 vec/scalar DataLoader가 BIG_ENDIAN int32로 읽으므로 호환을 위해
    // 4 bytes BE int32로 저장 (BpePrep.writeData와 동일 포맷). vocab 검증만 유지.
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
    perSource: Map<String, Long>,
    trainTokens: Int,
    valTokens: Int,
    seed: Long,
    valRatio: Float,
    filteredCount: Int,
    dedupCount: Int,
) {
    val sb = StringBuilder()
    sb.append("{\n")
    sb.append("  \"option\": \"C\",\n")
    sb.append("  \"seed\": $seed,\n")
    sb.append("  \"val_ratio\": $valRatio,\n")
    sb.append("  \"train_tokens\": $trainTokens,\n")
    sb.append("  \"val_tokens\": $valTokens,\n")
    sb.append("  \"v4_filtered_stories\": $filteredCount,\n")
    sb.append("  \"deduped_records\": $dedupCount,\n")
    sb.append("  \"source_tokens\": {\n")
    val entries = perSource.entries.toList()
    for ((i, e) in entries.withIndex()) {
        val comma = if (i < entries.size - 1) "," else ""
        sb.append("    \"${e.key}\": ${e.value}$comma\n")
    }
    sb.append("  }\n")
    sb.append("}\n")
    file.writeText(sb.toString())
}

private fun md5(s: String): String {
    val bytes = MessageDigest.getInstance("MD5").digest(s.toByteArray(Charsets.UTF_8))
    return buildString(bytes.size * 2) { for (b in bytes) append("%02x".format(b)) }
}
