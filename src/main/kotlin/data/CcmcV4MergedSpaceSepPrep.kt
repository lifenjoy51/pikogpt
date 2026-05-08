package data

import kotlinx.serialization.encodeToString
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
 * 공백을 별도 토큰으로 고정한 BPE prep.
 *
 * 기존 [CcmcV4MergedPrep]은 stage1 BPE meta(공백+단어 묶음, vocab=2000)를 재사용했지만,
 * 본 스크립트는 [CharBPE.splitSpaceAsToken] = true 로 **새 BPE를 학습**한다.
 * 결과:
 *   - 공백은 단일 ' ' (id 6 부근) 토큰으로 고정
 *   - BPE merge는 알파벳/숫자/기호만 묶음 → 같은 단어가 위치 무관 동일 토큰
 *
 * 입출력:
 *   v2-pro 4 .txt + v4 9 raw.jsonl → data/ccmc-v4-merged-spacesep/
 *   ├ train.bin  (uint16 little-endian)
 *   ├ val.bin
 *   ├ meta.json  (새로 학습한 stoi + merges)
 *   └ prep_manifest.json
 *
 * 사용법:
 *   ./gradlew runCcmcV4MergedSpaceSepPrep
 *   ./gradlew runCcmcV4MergedSpaceSepPrep \
 *       --args="<pikoData> <v4Base> <outDir> <vocabSize> <valRatio> <seed>"
 */
fun main(args: Array<String>) {
    val pikoData = args.getOrNull(0) ?: "data"
    val v4Base = args.getOrNull(1)
        ?: "${System.getProperty("user.home")}/works/llm-playground/data/processed/ccmc_v4_tinystories"
    val outDir = args.getOrNull(2) ?: "data/ccmc-v4-merged-spacesep"
    val vocabSize = args.getOrNull(3)?.toIntOrNull() ?: 2000
    val valRatio = (args.getOrNull(4)?.toFloatOrNull() ?: 0.05f).coerceIn(0.001f, 0.5f)
    val seed = args.getOrNull(5)?.toLongOrNull() ?: 42L

    println("=== CCMC v4-merged space-sep BPE prep ===")
    println("piko data : $pikoData")
    println("v4 base   : $v4Base")
    println("out dir   : $outDir")
    println("vocab size: $vocabSize")
    println("val ratio : $valRatio  seed: $seed")

    val parser = Json { ignoreUnknownKeys = true }

    // ── records 로드 ──────────────────────────────────────────────────────────
    val records = mutableListOf<SsRecord>()
    val v2Paths = listOf(
        "$pikoData/ccmc-v2-pro/stage1/train.txt",
        "$pikoData/ccmc-v2-pro/stage1/val.txt",
        "$pikoData/ccmc-v2-pro/stage2/train.txt",
        "$pikoData/ccmc-v2-pro/stage2/val.txt",
    )
    for (path in v2Paths) {
        val before = records.size
        records += loadV2LinesSs(File(path), source = "v2_pro", needsBosEos = false)
        println("  v2-pro [$path]: ${records.size - before} records")
    }
    val v4Folders = listOf(
        "flash", "flash_e2", "flash_e3", "flash_e4", "flash_e5", "flash_e6",
        "pro", "pro_e2", "pro_e3",
    )
    val banWords = setOf("arthopods", "adios", "amor", "asante", "gracias")
    var totalFiltered = 0
    for (folder in v4Folders) {
        val path = File("$v4Base/$folder/raw.jsonl")
        if (!path.exists()) {
            println("  [skip] v4 폴더 없음: $path"); continue
        }
        val before = records.size
        val (loaded, filtered) = loadV4JsonlSs(
            path, source = "v4_$folder", needsBosEos = true, banWords = banWords, parser = parser,
        )
        records += loaded
        totalFiltered += filtered
        println("  v4 [$folder]: ${records.size - before} records (filtered=$filtered)")
    }
    val seenKeys = HashSet<String>()
    val unique = records.filter { seenKeys.add(it.key) }
    val dedupCount = records.size - unique.size
    println("dedupe: ${records.size} → ${unique.size} (-$dedupCount)")

    // ── BPE 학습 (splitSpaceAsToken=true) ─────────────────────────────────────
    // BOS/EOS marker는 v4 wrap 단계에서 직접 id로 prepend되므로,
    // 학습 코퍼스에는 special tokens가 stage1/2 v2-pro 라인에서 자연스럽게 등장한다.
    val specialTokens = listOf(
        CharBPE.EOS_TOKEN, CharBPE.UNKNOWN_TOKEN, CharBPE.BOS_TOKEN,
        CharBPE.TURN_TOKEN, CharBPE.SEP_TOKEN,
    )
    val bpe = CharBPE(
        maxVocabSize = vocabSize,
        specialTokens = specialTokens,
        lowercase = false,
        useWordPreTokenize = false,
        standardBpeScoring = true,
        verbose = true,
        splitSpaceAsToken = true,
    )

    // 학습 텍스트 = 모든 record를 \n으로 join (v2-pro markers 그대로 포함, v4는 plain)
    println("\n=== BPE 학습 시작 (vocab=$vocabSize) ===")
    val trainText = StringBuilder()
    for (r in unique) { trainText.append(r.text); trainText.append("\n") }
    println("학습 코퍼스 char 수: ${trainText.length}")
    bpe.train(trainText.toString())
    println("=== BPE 학습 완료 ===")
    println("vocab=${bpe.getVocabSize()}, merges=${bpe.getMerges().size}")

    // ── stoi + merges 저장 ────────────────────────────────────────────────────
    val stoi = bpe.getStoi()
    val merges = bpe.getMerges().map { listOf(it.first, it.second) }
    val bosId = stoi[CharBPE.BOS_TOKEN] ?: error("BOS not learned")
    val eosId = stoi[CharBPE.EOS_TOKEN] ?: error("EOS not learned")
    val spaceId = stoi[" "] ?: error("space not in vocab — splitSpaceAsToken 동작 안 함")
    println("BOS=$bosId  EOS=$eosId  space=$spaceId")
    val sampleEnc = bpe.encode("the cat eats")
    println("sample encode 'the cat eats' = $sampleEnc → ${sampleEnc.map { id -> stoi.entries.firstOrNull { it.value == id }?.key }}")

    File(outDir).mkdirs()
    val meta = MetaInfo(
        vocabularySize = bpe.getVocabSize(),
        stringToIndex = stoi,
        indexToString = stoi.entries.associate { (k, v) -> v to k },
        merges = merges,
        lowercase = false,
        useWordPreTokenize = false,
        specialTokens = specialTokens,
    )
    val jsonOut = Json { prettyPrint = true; encodeDefaults = true }
    File(outDir, "meta.json").writeText(jsonOut.encodeToString(meta))

    // ── shuffle + split + encode ─────────────────────────────────────────────
    val shuffled = unique.toMutableList().apply { shuffle(Random(seed)) }
    val nVal = (shuffled.size * valRatio).toInt().coerceAtLeast(1)
    val valRec = shuffled.take(nVal)
    val trainRec = shuffled.drop(nVal)
    println("split: train=${trainRec.size} val=${valRec.size}")

    val sourceTokenCounts = LinkedHashMap<String, Long>()
    fun encodeRecord(r: SsRecord): List<Int> {
        val core = bpe.encode(r.text)
        return if (r.needsBosEos) {
            val out = ArrayList<Int>(core.size + 2)
            out.add(bosId); out.addAll(core); out.add(eosId)
            out
        } else core
    }
    fun encodeAll(recs: List<SsRecord>): List<Int> {
        val out = ArrayList<Int>(recs.size * 200)
        for ((i, r) in recs.withIndex()) {
            val toks = encodeRecord(r)
            out.addAll(toks)
            sourceTokenCounts.merge(r.source, toks.size.toLong()) { a, b -> a + b }
            if ((i + 1) % 5000 == 0) println("  encoded ${i + 1}/${recs.size}")
        }
        return out
    }
    println("encoding train..."); val trainTokens = encodeAll(trainRec)
    println("encoding val...");   val valTokens = encodeAll(valRec)
    println("token totals: train=${trainTokens.size}  val=${valTokens.size}")

    // bin write
    writeUint16BinSs(File(outDir, "train.bin"), trainTokens, bpe.getVocabSize())
    writeUint16BinSs(File(outDir, "val.bin"), valTokens, bpe.getVocabSize())

    // manifest
    val sb = StringBuilder()
    sb.append("{\n")
    sb.append("  \"option\": \"C-spacesep\",\n")
    sb.append("  \"split_space_as_token\": true,\n")
    sb.append("  \"seed\": $seed,\n")
    sb.append("  \"val_ratio\": $valRatio,\n")
    sb.append("  \"vocab_size\": ${bpe.getVocabSize()},\n")
    sb.append("  \"merges\": ${merges.size},\n")
    sb.append("  \"train_tokens\": ${trainTokens.size},\n")
    sb.append("  \"val_tokens\": ${valTokens.size},\n")
    sb.append("  \"v4_filtered_stories\": $totalFiltered,\n")
    sb.append("  \"deduped_records\": $dedupCount,\n")
    sb.append("  \"source_tokens\": {\n")
    val entries = sourceTokenCounts.entries.toList()
    for ((i, e) in entries.withIndex()) {
        val comma = if (i < entries.size - 1) "," else ""
        sb.append("    \"${e.key}\": ${e.value}$comma\n")
    }
    sb.append("  }\n}\n")
    File(outDir, "prep_manifest.json").writeText(sb.toString())

    println("=== prep done ===")
    println("  train.bin : ${trainTokens.size} tokens")
    println("  val.bin   : ${valTokens.size} tokens")
    println("  meta.json : new BPE (vocab=${bpe.getVocabSize()}, merges=${merges.size})")
    println("  source breakdown:")
    sourceTokenCounts.forEach { (s, c) -> println("    $s : $c tokens") }
}

// ── data class ─────────────────────────────────────────────────────────────────
private data class SsRecord(
    val source: String,
    val key: String,
    val text: String,
    val needsBosEos: Boolean,
)

// ── loaders (CcmcV4MergedPrep과 동일 로직 — internal helper로 분리 가능) ────
private fun loadV2LinesSs(file: File, source: String, needsBosEos: Boolean): List<SsRecord> {
    if (!file.exists()) return emptyList()
    return file.useLines { lines ->
        lines.mapNotNull { l ->
            val t = l.trim()
            if (t.isEmpty()) null else SsRecord(source, md5Ss(t), t, needsBosEos)
        }.toList()
    }
}

private fun loadV4JsonlSs(
    file: File,
    source: String,
    needsBosEos: Boolean,
    banWords: Set<String>,
    parser: Json,
): Pair<List<SsRecord>, Int> {
    val out = mutableListOf<SsRecord>()
    var filtered = 0
    val wordRe = Regex("[a-z][a-z'-]*")
    file.useLines { lines ->
        for (line in lines) {
            val t = line.trim()
            if (t.isEmpty() || !t.startsWith("{")) continue
            val obj = try { parser.parseToJsonElement(t).jsonObject } catch (_: Exception) { continue }
            val text = obj["text"]?.jsonPrimitive?.contentOrNull?.trim() ?: continue
            if (text.isEmpty()) continue
            val lower = text.lowercase()
            var banned = false
            for (m in wordRe.findAll(lower)) if (m.value in banWords) { banned = true; break }
            if (banned) { filtered++; continue }
            out += SsRecord(source, md5Ss(text), text, needsBosEos)
        }
    }
    return out to filtered
}

private fun writeUint16BinSs(file: File, tokens: List<Int>, vocabSize: Int) {
    // vec/scalar DataLoader 호환을 위해 BIG_ENDIAN int32로 저장 (BpePrep과 동일).
    file.parentFile?.mkdirs()
    RandomAccessFile(file, "rw").use { raf ->
        raf.setLength(0)
        val buf = ByteBuffer.allocate(tokens.size * 4).order(ByteOrder.BIG_ENDIAN)
        for (t in tokens) {
            require(t in 0 until vocabSize) { "token $t out of vocab $vocabSize" }
            buf.putInt(t)
        }
        raf.write(buf.array())
    }
}

private fun md5Ss(s: String): String {
    val bytes = MessageDigest.getInstance("MD5").digest(s.toByteArray(Charsets.UTF_8))
    return buildString(bytes.size * 2) { for (b in bytes) append("%02x".format(b)) }
}
