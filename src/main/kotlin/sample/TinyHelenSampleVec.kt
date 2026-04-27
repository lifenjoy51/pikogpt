package sample

import kotlinx.coroutines.runBlocking
import java.io.File
import vec.VecSampler

/**
 * TinyHelen 벡터 체크포인트 자동 샘플링.
 *
 * `model/vec/` 아래에서 가장 최근 수정된 체크포인트 디렉토리(`checkpoint.json` 존재)를 찾아
 * 여러 프롬프트로 생성한다. 인자가 있으면 해당 경로를 직접 사용.
 */
fun main(args: Array<String>) = runBlocking {
    val checkpointDir = args.getOrNull(0)?.let { File(it) }
        ?: findLatestVecCheckpoint()
        ?: error("벡터 체크포인트 없음: model/vec/ 아래를 확인하세요")

    println("=== 벡터 백엔드 샘플링 체크포인트: ${checkpointDir.absolutePath} ===")

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

// model/vec/*/*/ 중 checkpoint.json이 있는 디렉토리 중 가장 최근 수정본 반환.
private fun findLatestVecCheckpoint(): File? {
    val vecRoot = File("model/vec")
    if (!vecRoot.exists()) return null
    return vecRoot.listFiles()
        ?.filter { it.isDirectory }
        ?.flatMap { paramDir ->
            paramDir.listFiles()
                ?.filter { it.isDirectory && File(it, "checkpoint.json").exists() }
                ?.toList() ?: emptyList()
        }
        ?.maxByOrNull { File(it, "checkpoint.json").lastModified() }
}
