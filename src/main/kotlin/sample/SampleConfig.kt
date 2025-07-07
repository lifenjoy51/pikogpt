package sample

import kotlinx.serialization.Serializable

// 샘플링 설정
@Serializable
data class SampleConfig(
    val initFrom: String = "resume", // 'resume' 또는 'scratch'
    val outDir: String = "out-shakespeare-char",
    val start: String = "\n", // 시작 프롬프트 또는 "FILE:prompt.txt"
    val numSamples: Int = 10,
    val maxNewTokens: Int = 500,
    val temperature: Double = 0.8,
    val topK: Int = 200,
    val seed: Int = 1337
)
