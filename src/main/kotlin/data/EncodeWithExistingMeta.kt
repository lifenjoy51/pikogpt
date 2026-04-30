package data

import kotlinx.serialization.json.Json
import java.io.File

/**
 * 이미 학습된 vocab(`meta.json`)으로 다른 디렉터리의 `train.txt`/`val.txt`를 인코딩.
 *
 * 두 단계 학습(BASE + IT)에서 두 코퍼스가 **같은 token ID 공간**을 공유하도록,
 * 합본으로 BPE를 한 번 학습한 뒤 각 split을 같은 vocab으로 인코딩하는 데 사용.
 *
 * 사용법:
 *   ./gradlew runEncodeWithExistingMeta --args="<meta.json 경로> <인코딩 대상 디렉터리>"
 *
 * 인자:
 *   args[0] = vocab을 담은 meta.json 경로
 *   args[1] = train.txt(필수) 와 val.txt(선택) 가 있는 디렉터리. 같은 디렉터리에
 *             train.bin / val.bin 출력 + meta.json 복사.
 */
fun main(args: Array<String>) {
    require(args.size >= 2) {
        "사용법: runEncodeWithExistingMeta <meta.json 경로> <인코딩 대상 디렉터리>"
    }
    val metaPath = args[0]
    val targetDir = File(args[1])

    require(File(metaPath).exists()) { "meta.json 없음: $metaPath" }
    require(targetDir.isDirectory) { "대상 디렉터리가 존재하지 않음: ${targetDir.absolutePath}" }

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

    encodeIfExists(bpe, File(targetDir, "train.txt"), File(targetDir, "train.bin"))
    encodeIfExists(bpe, File(targetDir, "val.txt"), File(targetDir, "val.bin"))

    // meta.json 복사 — VecTrainer.readVocabSize가 dataPath/meta.json을 읽으므로.
    val destMeta = File(targetDir, "meta.json")
    File(metaPath).copyTo(destMeta, overwrite = true)
    println("meta.json 복사 완료: ${destMeta.absolutePath}")
}

private fun encodeIfExists(bpe: SimpleBPE, src: File, dst: File) {
    if (!src.exists()) {
        println("스킵: ${src.absolutePath} 없음")
        return
    }
    val text = src.readText()
    val tokens = bpe.encode(text)
    StoriesBPEPrep.writeData(tokens, dst)
    println("인코딩 완료: ${src.name} (${text.length} chars) → ${dst.name} (${tokens.size} tokens)")
}
