package sample

import kotlinx.coroutines.runBlocking
import turbo.TurboSampler
import java.io.File

/**
 * ELF vs pikogpt 비교 실험용 샘플링 entry.
 *
 * 사용:
 *   ./gradlew runCompareSampler --args="[N] [maxNewTokens] [temperature] [topK]
 *                                       [exp=<expName>] [data=<datasetDir>]"
 *
 * 기본값:
 *   N=8, maxNewTokens=40, temperature=0.9, topK=비활성
 *   exp=compare-vs-elf, data=ccmc-all-v2048-v6
 *
 * ckpt 디렉터리: `model/<data>/<exp>/v<latest>/`
 *
 * 프롬프트는 `<|bos|>` 한 토큰만 — 무조건 생성.
 */
fun main(args: Array<String>) = runBlocking {
    var numSamples = 8
    var maxNewTokens = 40
    var temperature = 0.9f
    var topK = 0
    var expName = "compare-vs-elf"
    var dataset = "ccmc-all-v2048-v7"
    var prompt = "<|bos|>"
    var ckptOverride: String? = null

    val positional = mutableListOf<String>()
    for (a in args) {
        when {
            a.startsWith("exp=") -> expName = a.substringAfter("exp=")
            a.startsWith("data=") -> dataset = a.substringAfter("data=")
            a.startsWith("prompt=") -> prompt = a.substringAfter("prompt=").replace('_', ' ')
            a.startsWith("ckpt=") -> ckptOverride = a.substringAfter("ckpt=")
            else -> positional += a
        }
    }
    positional.getOrNull(0)?.toIntOrNull()?.let { numSamples = it }
    positional.getOrNull(1)?.toIntOrNull()?.let { maxNewTokens = it }
    positional.getOrNull(2)?.toFloatOrNull()?.let { temperature = it }
    positional.getOrNull(3)?.toIntOrNull()?.let { topK = it }

    val ckptDir = if (ckptOverride != null) {
        val explicit = File("model/$dataset/$expName/$ckptOverride")
        require(explicit.exists()) { "ckpt 디렉터리 없음: ${explicit.absolutePath}" }
        explicit
    } else {
        latestVersionDir(File("model"), dataset, expName)
            ?: error("ckpt 디렉터리 없음: model/$dataset/$expName/")
    }
    println("ckpt 로드: ${ckptDir.absolutePath}")

    val config = SampleConfig(
        modelDirectoryPath = ckptDir.absolutePath,
        numberOfSamples = numSamples,
        maximumNewTokens = maxNewTokens,
        samplingTemperature = temperature,
        topKFilteringSize = topK,
        randomSeed = System.currentTimeMillis().toInt(),
    )

    val sampler = TurboSampler(config)
    val outputs = sampler.generate(prompt)

    println(
        "샘플링: numSamples=$numSamples maxNewTokens=$maxNewTokens " +
            "temperature=$temperature topK=$topK exp=$expName prompt='$prompt'"
    )
    println("-".repeat(60))
    outputs.forEachIndexed { i, line ->
        println("[$i] $line")
    }
}

private fun latestVersionDir(modelDir: File, dataset: String, expName: String): File? {
    val expDir = File(File(modelDir, dataset), expName)
    if (!expDir.exists()) return null
    return expDir.listFiles { f -> f.isDirectory && f.name.matches(Regex("v\\d+")) }
        ?.maxByOrNull { it.name.substring(1).toInt() }
}
