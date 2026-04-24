package sample

import kotlinx.coroutines.runBlocking
import java.io.File
import vec.Sampler as VecSampler

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

    val config = SampleConfig(
        modelDirectoryPath = checkpointDir.absolutePath,
        numberOfSamples = 2,
        maximumNewTokens = 120,
        samplingTemperature = 0.8f,
        topKFilteringSize = 40,
    )
    val sampler = VecSampler(config)

    for (prompt in prompts) {
        val outputs = sampler.generate(prompt)
        println("\n=== Prompt: $prompt ===")
        outputs.forEachIndexed { i, line ->
            println("[샘플 ${i + 1}]")
            println(line)
        }
    }
}
