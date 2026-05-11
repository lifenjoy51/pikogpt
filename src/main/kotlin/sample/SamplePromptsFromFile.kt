package sample

import data.MetaInfo
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import java.io.File
import turbo.TurboSampler

/**
 * Turbo 체크포인트 + 프롬프트 파일로 커스텀 샘플링.
 *
 * 사용: `./gradlew runSamplePromptsFromFile --args="<ckpt-dir> <prompts.txt> [with-turn] [--temp=X]"`
 *   - `<ckpt-dir>`: `checkpoint.json` + `model_weights.bin` + `meta.json` 있는 경로
 *   - `<prompts.txt>`: 한 줄에 하나의 프롬프트 (빈 줄 무시)
 *   - `with-turn` (선택): meta.json에 `<|turn|>` 토큰이 있고 instruction-tune 모델처럼 사용자 발화 종료
 *      신호로 turn 추가가 필요할 때 명시. 기본은 prompt를 **그대로** 모델에 전달 (raw mode).
 *   - `--temp=X` (선택): single temperature 강제. 미지정 시 default `temps` 적용.
 *
 * **중요**: 이전 버전은 `<|turn|>`이 meta에 있으면 자동으로 prompt 끝에 추가했음.
 * dict/wordstart 같이 `<|turn|>`이 vocab에는 있지만 prompt 직후 추가하면 안 되는 데이터셋에서
 * 잘못된 input이 모델에 들어가 mode collapse처럼 보이는 현상을 일으켰음. 이 동작은 명시적 옵션화.
 */
fun main(args: Array<String>) = runBlocking {
    require(args.size >= 2) { "사용: <ckpt-dir> <prompts.txt> [with-turn] [--temp=X]" }
    val tempArg = args.firstOrNull { it.startsWith("--temp=") }?.removePrefix("--temp=")?.toFloatOrNull()
    val positional = args.filter { !it.startsWith("--") }
    val checkpointDir = File(positional[0])
    val promptsFile = File(positional[1])
    require(checkpointDir.exists()) { "ckpt 경로 없음: ${checkpointDir.absolutePath}" }
    require(promptsFile.exists()) { "prompts 파일 없음: ${promptsFile.absolutePath}" }

    val prompts = promptsFile.readLines().map { it.trimEnd() }.filter { it.isNotBlank() }
    println("=== Turbo 백엔드 샘플링 ckpt: ${checkpointDir.absolutePath} ===")
    println("프롬프트 수: ${prompts.size}")

    val metaFile = File(checkpointDir, "meta.json")
    val turnId: Int? = if (metaFile.exists()) {
        val parser = Json { ignoreUnknownKeys = true }
        val meta = parser.decodeFromString<MetaInfo>(metaFile.readText())
        meta.stringToIndex["<|turn|>"]
    } else null
    val stopIds = mutableListOf(0).also { if (turnId != null) it.add(turnId) }
    if (turnId != null) println("stop tokens: EOS=0, <|turn|>=$turnId")
    else println("stop tokens: EOS=0")

    // appendTurn은 **명시적 with-turn 옵션이 있을 때만** 활성. 기본은 raw prompt 그대로.
    val appendTurn = turnId != null && positional.getOrNull(2) == "with-turn"
    if (appendTurn) println("with-turn 모드: prompt 끝에 <|turn|> 토큰 자동 추가")
    else println("raw 모드: prompt를 그대로 모델에 전달 (with-turn 옵션으로 활성화)")

    val temps: List<Float> = if (tempArg != null) listOf(tempArg) else listOf(0.0f, 0.5f, 0.99f)
    println("temperatures=$temps")

    for (temp in temps) {
        val config = SampleConfig(
            modelDirectoryPath = checkpointDir.absolutePath,
            numberOfSamples = 1,
            maximumNewTokens = 120,
            samplingTemperature = temp,
            topKFilteringSize = 40,
            topProbabilityThreshold = 0.95f,
            repetitionPenalty = 1.15f,
            stopTokenIds = stopIds,
        )
        val sampler = TurboSampler(config)
        println("\n##### temperature=$temp #####")
        for (prompt in prompts) {
            println("\n=== Prompt: $prompt ===")
            val promptIdsBase = sampler.encodeText(prompt).toMutableList()
            if (appendTurn) promptIdsBase.add(turnId!!)
            val (_, response) = sampler.continueOne(promptIdsBase.toIntArray())
            println("[샘플 1]")
            println(response.trim())
        }
    }
}
