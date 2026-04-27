package sample

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import java.io.File

/**
 * TinyHelen 학습 후 자동 샘플링.
 *
 * `model/` 아래에서 가장 최근에 갱신된 체크포인트 디렉토리를 찾아 여러 프롬프트로 생성.
 * 인자가 있으면 해당 경로를 직접 사용.
 */
fun main(args: Array<String>) = runBlocking {
    val checkpointDir = args.getOrNull(0)?.let { File(it) } ?: findLatestCheckpoint()
    ?: error("체크포인트 디렉토리를 찾을 수 없습니다. model/ 아래를 확인하세요.")

    println("=== 샘플링 대상 체크포인트: ${checkpointDir.absolutePath} ===")

    val prompts = listOf(
        "Once upon a time",
        "The little girl",
        "In the morning",
        "He opened the book",
        "She said,",
    )

    withContext(Dispatchers.Default) {
        val config = SampleConfig(
            modelDirectoryPath = checkpointDir.absolutePath,
            numberOfSamples = 2,
            maximumNewTokens = 80,
            samplingTemperature = 0.8f,
            topKFilteringSize = 40,
        )
        val sampler = ScalarSampler(config)

        prompts.forEach { prompt ->
            val result = sampler.generateText(prompt)
            println("\n=== Prompt: ${result.prompt} ===")
            result.results.forEachIndexed { i, line ->
                println("[샘플 ${i + 1}]")
                println(line)
            }
        }
    }
}

// model 루트의 모든 2단 하위 디렉토리 중 checkpoint.json이 있는 곳에서 가장 최근 수정본 반환.
private fun findLatestCheckpoint(): File? {
    val modelRoot = File("model")
    if (!modelRoot.exists()) return null
    return modelRoot.listFiles()
        ?.filter { it.isDirectory }
        ?.flatMap { paramDir ->
            paramDir.listFiles()
                ?.filter { it.isDirectory && File(it, "checkpoint.json").exists() }
                ?.toList() ?: emptyList()
        }
        ?.maxByOrNull { File(it, "checkpoint.json").lastModified() }
}
