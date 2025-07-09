package gpt

// 사용 예시
fun main() {
    // 작은 설정으로 모델 생성
    val config = GPTConfig(
        blockSize = 32,
        vocabSize = 100,
        nLayer = 2,
        nHead = 2,
        nEmbd = 64,
        bias = true
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