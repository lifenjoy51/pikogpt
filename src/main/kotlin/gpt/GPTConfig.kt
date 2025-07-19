package gpt

import kotlinx.serialization.Serializable

// GPT 설정을 위한 데이터 클래스
@Serializable
data class GPTConfig(
    val blockSize: Int,
    val vocabSize: Int,
    val nLayer: Int,
    val nHead: Int,
    val nEmbd: Int,
    val bias: Boolean,
    val dropout: Float
)