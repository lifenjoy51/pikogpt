package gpt

import kotlinx.serialization.Serializable

// GPT 설정을 위한 데이터 클래스
@Serializable
data class GPTConfig(
    val blockSize: Int = 1024,
    val vocabSize: Int = 50304,
    val nLayer: Int = 12,
    val nHead: Int = 12,
    val nEmbd: Int = 768,
    val bias: Boolean = true,
    val dropout: Float = 0.1f
)