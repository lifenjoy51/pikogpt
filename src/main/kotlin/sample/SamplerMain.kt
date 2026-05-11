package sample

import kotlinx.coroutines.runBlocking
import java.io.File

/**
 * Scalar 백엔드 quickstart 샘플링 진입점.
 *
 * MiniTrainerMain으로 학습한 ckpt를 로드해 알파벳 패턴이 어느 정도 학습됐는지 텍스트 생성으로 확인.
 *
 * 사용법:
 *   ./gradlew runSampler
 *     # → model/alphabet/main 아래 가장 큰 v 번호 디렉토리 자동 검색
 *
 *   ./gradlew runSampler --args="model/alphabet/main/v0001"
 *     # → 명시 ckpt 디렉토리 사용
 *
 * 기대 출력:
 *   학습 후반부의 ckpt에서는 "the cat ran"처럼 자주 나오는 단어 시퀀스가 부분적으로
 *   재현됩니다. 작은 모델이라 완벽한 문장은 아니지만 알파벳·음절 패턴은 살아 있어야 합니다.
 *
 * 자세한 한 흐름 가이드: docs/scalar-quickstart.md
 */
fun main(args: Array<String>) = runBlocking {
    val ckptPath = args.firstOrNull() ?: findLatestCheckpoint("model/alphabet/main")
    require(File(ckptPath).exists()) {
        "ckpt 디렉토리가 없습니다: $ckptPath\n" +
            "먼저 './gradlew runMiniTrainer'로 학습하거나 인자로 ckpt 경로를 지정하세요."
    }

    val config = SampleConfig(
        modelDirectoryPath = ckptPath,
        numberOfSamples = 1,
        maximumNewTokens = 15,
        samplingTemperature = 0.8f,
        topKFilteringSize = 20,
        topProbabilityThreshold = 1.0f,   // nucleus off (default)
        repetitionPenalty = 1.0f,         // off (default)
    )
    val sampler = ScalarSampler(config)

    // wordstart-prefix 학습 검증: a~z 단일 글자 프롬프트로 첫 글자 conditioning 확인.
    // BOS/EOS 데이터에선 <|bos|>+letter로 "새 단어 시작" 컨텍스트를 줌 (CharBPE가 longest-match로 인식).
    val prompts = ('a'..'z').map { "<|bos|>$it" }

    for (prompt in prompts) {
        println("\n=== prompt: '$prompt' ===")
        val result = sampler.generateText(prompt)
        result.results.forEachIndexed { i, line -> println("  [${i + 1}] $line") }
    }
}

/**
 * `${baseDir}/v0001`, `v0002`, ... 중 가장 큰 번호의 디렉토리 경로를 반환.
 * 없으면 IllegalStateException.
 */
private fun findLatestCheckpoint(baseDir: String): String {
    val base = File(baseDir)
    require(base.exists() && base.isDirectory) {
        "$baseDir 디렉토리가 없습니다. 먼저 './gradlew runMiniTrainer'로 학습하세요."
    }
    val versionRegex = Regex("""v(\d{4})""")
    val latest = base.listFiles()
        ?.filter { it.isDirectory && versionRegex.matches(it.name) }
        ?.maxByOrNull { versionRegex.matchEntire(it.name)!!.groupValues[1].toInt() }
        ?: error(
            "$baseDir 아래에 v0001 형식의 ckpt 디렉토리가 없습니다. " +
                "먼저 './gradlew runMiniTrainer'로 학습하세요."
        )
    println("# 자동 검색: ${latest.absolutePath}")
    return latest.absolutePath
}
