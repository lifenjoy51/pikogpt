package sample

import kotlinx.serialization.Serializable

// 샘플링 설정
@Serializable
data class SampleConfig(
    val initFrom: String = "resume", // 'resume' 또는 'scratch'
    val modelDir: String = "model",
    val numSamples: Int = 10,
    val maxNewTokens: Int = 50,
    val temperature: Float = 0.8f,
    val topK: Int = 100,
    val seed: Int = 1337
)
