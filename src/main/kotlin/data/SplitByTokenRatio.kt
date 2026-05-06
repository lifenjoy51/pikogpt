package data

import kotlinx.serialization.json.Json
import java.io.File

/**
 * record-per-line 텍스트를 BPE 토큰 수 기준으로 train/val에 분할.
 *
 * 사용법:
 *   ./gradlew runSplitByTokenRatio --args="<meta.json> <input.txt> <out-dir> [ratio] [seed]"
 *
 * 인자:
 *   args[0] = vocab을 담은 meta.json 경로 (encode 기준)
 *   args[1] = 입력 텍스트 — 한 줄 = 한 record (`<|bos|>...<|eos|>` 묶음, 빈 줄 무시)
 *   args[2] = 출력 디렉터리 — train.txt / val.txt 생성
 *   args[3] = train ratio (기본 0.9)
 *   args[4] = shuffle seed (기본 42)
 *
 * 동작:
 *   1) meta.json으로 SimpleBPE 복원
 *   2) 각 record를 encode → 토큰 수 측정
 *   3) seed 기반 shuffle 후 누적 토큰이 ratio 도달할 때까지 train, 나머지 val
 *   4) train.txt / val.txt를 record-per-line으로 출력
 *   5) 분할 결과 통계 stdout 출력
 */
fun main(args: Array<String>) {
    require(args.size >= 3) {
        "사용법: runSplitByTokenRatio <meta.json> <input.txt> <out-dir> [ratio] [seed]"
    }
    val metaPath = args[0]
    val inputFile = File(args[1])
    val outDir = File(args[2])
    val ratio = args.getOrNull(3)?.toDoubleOrNull() ?: 0.9
    val seed = args.getOrNull(4)?.toLongOrNull() ?: 42L

    require(File(metaPath).exists()) { "meta.json 없음: $metaPath" }
    require(inputFile.exists()) { "input.txt 없음: ${inputFile.absolutePath}" }
    require(outDir.isDirectory) { "출력 디렉터리 없음: ${outDir.absolutePath}" }
    require(ratio in 0.0..1.0) { "ratio 범위 [0,1]: $ratio" }

    val meta = Json { ignoreUnknownKeys = true }
        .decodeFromString(MetaInfo.serializer(), File(metaPath).readText())
    val merges = meta.merges.map { it[0] to it[1] }
    val bpe = SimpleBPE(
        maxVocabSize = meta.vocabularySize,
        specialTokens = meta.specialTokens,
        lowercase = meta.lowercase,
        useWordPreTokenize = meta.useWordPreTokenize,
        standardBpeScoring = true,
        verbose = false,
    )
    bpe.restore(meta.stringToIndex, merges)
    println("Vocab 복원 완료: size=${bpe.getVocabSize()}, merges=${merges.size}")

    val records = inputFile.readLines().map { it.trim() }.filter { it.isNotEmpty() }
    println("입력 records: ${records.size} (from ${inputFile.name})")

    val tokenCounts = IntArray(records.size)
    var totalTokens = 0L
    for ((i, rec) in records.withIndex()) {
        val n = bpe.encode(rec).size
        tokenCounts[i] = n
        totalTokens += n
    }
    println("총 토큰 수: $totalTokens, record당 평균 ${"%.1f".format(totalTokens.toDouble() / records.size)}")

    val indices = records.indices.toMutableList()
    indices.shuffle(kotlin.random.Random(seed))
    val targetTrainTokens = (totalTokens * ratio).toLong()

    val trainIdx = mutableListOf<Int>()
    val valIdx = mutableListOf<Int>()
    var trainTokens = 0L
    for (i in indices) {
        if (trainTokens < targetTrainTokens) {
            trainIdx.add(i)
            trainTokens += tokenCounts[i]
        } else {
            valIdx.add(i)
        }
    }
    val valTokens = totalTokens - trainTokens

    val trainOut = File(outDir, "train.txt")
    val valOut = File(outDir, "val.txt")
    trainOut.bufferedWriter().use { w ->
        for (i in trainIdx) {
            w.write(records[i]); w.newLine()
        }
    }
    valOut.bufferedWriter().use { w ->
        for (i in valIdx) {
            w.write(records[i]); w.newLine()
        }
    }

    println("=== 분할 결과 (seed=$seed, target ratio=$ratio) ===")
    println("train: ${trainIdx.size} records, $trainTokens tokens (${"%.2f%%".format(100.0 * trainTokens / totalTokens)})")
    println("val:   ${valIdx.size} records, $valTokens tokens (${"%.2f%%".format(100.0 * valTokens / totalTokens)})")
    println("출력: ${trainOut.absolutePath}")
    println("출력: ${valOut.absolutePath}")
}
