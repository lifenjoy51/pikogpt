package sample

import data.MetaInfo
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import turbo.TurboSampler
import java.io.File

/**
 * v4-merged-spacesep 3M 모델 (best ckpt v0022, val 1.84) 재샘플링.
 *
 *   - greedy(T=0)에서 보였던 "in the park." mode collapse 검증용
 *   - T=0.8 + topK=40 + topP=0.9 + repetitionPenalty=1.15로 다양성 회복 확인
 *   - 동일 prompt 10개 × 3 sample
 */
fun main(args: Array<String>) = runBlocking {
    // 인자 없으면 새 경로 default (`expName="3m"` 진입점 + 첫 ckpt). 인자가 있으면 그 경로 사용.
    val ckptPath = args.firstOrNull() ?: "model/ccmc-v4-merged-spacesep-4k/3m/v0001"
    val ckptDir = File(ckptPath)
    require(ckptDir.exists()) { "ckpt 경로 없음: ${ckptDir.absolutePath}" }

    val metaFile = File(ckptDir, "meta.json")
    val parser = Json { ignoreUnknownKeys = true }
    val meta = parser.decodeFromString<MetaInfo>(metaFile.readText())
    // QA 모델 진단: stop=[0(eos)]만 — turn은 답 끝 신호로 학습됐으므로 stop에 두면 즉시 종료.
    val stopIds = listOf(0)

    val config = SampleConfig(
        modelDirectoryPath = ckptDir.absolutePath,
        numberOfSamples = 3,
        maximumNewTokens = 100,
        samplingTemperature = 0.8f,
        topKFilteringSize = 40,
        topProbabilityThreshold = 0.9f,
        repetitionPenalty = 1.15f,
        stopTokenIds = stopIds,
        randomSeed = 51,
    )

    // v2-pro stage2 학습 형식 그대로: <|bos|>Q?<|turn|> — 모델이 답을 이어 생성.
    // stop=[0(eos), 3(turn)] 유지 → single-turn 답 받음.
    val prompts = listOf(
        "<|bos|>What is in the park?<|turn|>",
        "<|bos|>Where is the cat?<|turn|>",
        "<|bos|>Why is the boy happy?<|turn|>",
        "<|bos|>How do you eat a cake?<|turn|>",
        "<|bos|>When does the dog run?<|turn|>",
        "<|bos|>Who is in the kitchen?<|turn|>",
        "<|bos|>What can you see in a tree?<|turn|>",
        "<|bos|>Is the water cold?<|turn|>",
        "<|bos|>Do you like to play?<|turn|>",
        "<|bos|>Can we go home?<|turn|>",
    )

    println("=== v0022 재샘플링 (T=${config.samplingTemperature}, topK=${config.topKFilteringSize}, topP=${config.topProbabilityThreshold}, repPenalty=${config.repetitionPenalty}) ===")
    println("stop tokens: $stopIds")
    println()

    val sampler = TurboSampler(config)
    for (prompt in prompts) {
        println("=== '$prompt' ===")
        val outputs = sampler.generate(prompt)
        outputs.forEachIndexed { i, s -> println("  [${i + 1}] ${s.trim()}") }
        println()
    }
}
