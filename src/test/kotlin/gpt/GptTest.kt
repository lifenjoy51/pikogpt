package gpt

import kotlin.test.Test

class GptTest {
    @Test
    fun main() {
        // 작은 설정으로 모델 생성
        val config = GPTConfig(
            maxSequenceLength = 32,
            vocabularySize = 100,
            numberOfLayers = 1,
            numberOfAttentionHeads = 1,
            embeddingDimension = 16,
            useBias = true,
            dropoutProbability = 0.1f
        )

        val model = PikoGPT(config)
        println("모델 생성 완료!")
        println("파라미터 수: ${model.parameters().size}")

        // 간단한 입력으로 테스트
        val input = intArrayOf(1, 2, 3, 4, 5)
        val output = model.forward(input)
        println("입력 시퀀스 길이: ${input.size}")
        println("출력 $output")
        println("출력 shape: ${output.size} x ${output[0].size}")
    }
}