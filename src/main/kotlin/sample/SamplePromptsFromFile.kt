package sample

import data.MetaInfo
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import java.io.File
import vec.VecSampler

/**
 * 벡터 체크포인트 + 프롬프트 파일로 커스텀 샘플링.
 *
 * 사용: `./gradlew runSamplePromptsFromFile --args="<ckpt-dir> <prompts.txt>"`
 *   - ckpt-dir: `checkpoint.json` + `model_weights.bin` + `meta.json` 있는 경로
 *   - prompts.txt: 한 줄에 하나의 프롬프트 (빈 줄 무시)
 *
 * 출력은 프롬프트별 샘플을 JSON 비슷한 구조로 stdout에 찍어 다음 단계 리포트 생성에 파이프하기 쉽게.
 */
fun main(args: Array<String>) = runBlocking {
    require(args.size >= 2) { "사용: <ckpt-dir> <prompts.txt>" }
    val checkpointDir = File(args[0])
    val promptsFile = File(args[1])
    require(checkpointDir.exists()) { "ckpt 경로 없음: ${checkpointDir.absolutePath}" }
    require(promptsFile.exists()) { "prompts 파일 없음: ${promptsFile.absolutePath}" }

    val prompts = promptsFile.readLines().map { it.trim() }.filter { it.isNotEmpty() }
    println("=== 벡터 백엔드 샘플링 ckpt: ${checkpointDir.absolutePath} ===")
    println("프롬프트 수: ${prompts.size}")

    // meta.json에 <|turn|>이 있으면 그 id를 stop 조건에 자동 추가 → single-turn 응답
    val metaFile = File(checkpointDir, "meta.json")
    val turnId: Int? = if (metaFile.exists()) {
        val parser = Json { ignoreUnknownKeys = true }
        val meta = parser.decodeFromString<MetaInfo>(metaFile.readText())
        meta.stringToIndex["<|turn|>"]
    } else null
    val stopIds = mutableListOf(0).also { if (turnId != null) it.add(turnId) }
    if (turnId != null) println("stop tokens: EOS=0, <|turn|>=$turnId (single-turn 응답 모드)")
    else println("stop tokens: EOS=0")

    val config = SampleConfig(
        modelDirectoryPath = checkpointDir.absolutePath,
        numberOfSamples = 2,
        maximumNewTokens = 120,
        samplingTemperature = 0.8f,
        topKFilteringSize = 40,
        topProbabilityThreshold = 0.95f,  // top-k 위에 nucleus 추가 — 보수적 다양성 컷
        repetitionPenalty = 1.15f,         // 반복 차단 (mode collapse 완화)
        stopTokenIds = stopIds,
    )
    val sampler = VecSampler(config)

    // turn 토큰이 있는 모델에서는 prompt를 "사용자 turn 종료"로 보고 다음 turn(=응답)을 생성하도록
    // prompt 끝에 <|turn|>을 자동 추가. 이렇게 안 하면 따옴표로 닫힌 prompt가 곧바로
    // <|turn|>을 예측해 빈 응답이 나옴.
    // dict 같은 turn 없는 모델은 args[2]="no-turn"으로 비활성화.
    val appendTurn = turnId != null && args.getOrNull(2) != "no-turn"

    for (prompt in prompts) {
        println("\n=== Prompt: $prompt ===")
        val promptIdsBase = sampler.encodeText(prompt).toMutableList()
        if (appendTurn) promptIdsBase.add(turnId!!)
        val numSamples = config.numberOfSamples
        for (i in 0 until numSamples) {
            val (_, response) = sampler.continueOne(promptIdsBase.toIntArray())
            println("[샘플 ${i + 1}]")
            println(response.trim())
        }
    }
}
