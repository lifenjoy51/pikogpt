package sample

import kotlinx.coroutines.runBlocking
import java.io.File
import turbo.TurboSampler

/**
 * TinyHelen turbo 체크포인트 자동 샘플링.
 *
 * `model/<datasetName>/turbo/` 아래에서 가장 최근 수정된 체크포인트(`checkpoint.json` 존재)를
 * 찾아 여러 프롬프트로 생성. 인자가 있으면 해당 경로를 직접 사용.
 */
fun main(args: Array<String>) = runBlocking {
    val checkpointDir = args.getOrNull(0)?.let { File(it) }
        ?: findLatestTurboCheckpoint()
        ?: error("turbo 체크포인트 없음: model/*/turbo/ 아래를 확인하세요")

    println("=== turbo 백엔드 샘플링 체크포인트: ${checkpointDir.absolutePath} ===")

    val prompts = listOf(
        "Once upon a time",
        "The little girl",
        "In the morning",
        "He opened the book",
        "She said,",
    )

    val config = SampleConfig(
        modelDirectoryPath = checkpointDir.absolutePath,
        numberOfSamples = 2,
        maximumNewTokens = 80,
        samplingTemperature = 0.8f,
        topKFilteringSize = 40,
    )
    val sampler = TurboSampler(config)

    for (prompt in prompts) {
        val outputs = sampler.generate(prompt)
        println("\n=== Prompt: $prompt ===")
        outputs.forEachIndexed { i, line ->
            println("[샘플 ${i + 1}]")
            println(line)
        }
    }
}

/** model/<datasetName>/turbo/<paramCount>/<v...>/ 트리에서 checkpoint.json 가장 최근 수정 디렉토리. */
private fun findLatestTurboCheckpoint(): File? {
    val modelRoot = File("model")
    if (!modelRoot.exists()) return null
    return modelRoot.listFiles()
        ?.filter { it.isDirectory }
        ?.flatMap { datasetDir ->
            val turboRoot = File(datasetDir, "turbo")
            if (!turboRoot.exists()) emptyList()
            else turboRoot.listFiles()
                ?.filter { it.isDirectory }
                ?.flatMap { paramDir ->
                    paramDir.listFiles()
                        ?.filter { it.isDirectory && File(it, "checkpoint.json").exists() }
                        ?.toList() ?: emptyList()
                } ?: emptyList()
        }
        ?.maxByOrNull { File(it, "checkpoint.json").lastModified() }
}
