package sample

import org.junit.jupiter.api.Assertions.*

class SamplerTest {


    // 다양한 설정으로 샘플링하는 함수들
    fun sampleShakespeare() {
        val config = SampleConfig(
            initFrom = "resume",
            outDir = "out-shakespeare-char",
            start = "ROMEO: ",
            numSamples = 3,
            maxNewTokens = 200,
            temperature = 0.8,
            topK = 40
        )

        val sampler = Sampler(config)
        sampler.sample()
    }

    fun sampleWithDifferentTemperatures() {
        val temperatures = listOf(0.5, 0.8, 1.0, 1.2)

        for (temp in temperatures) {
            println("\n========== Temperature: $temp ==========")
            val config = SampleConfig(
                start = "To be or not to be",
                numSamples = 1,
                maxNewTokens = 100,
                temperature = temp,
                topK = 0 // top-k 비활성화
            )

            val sampler = Sampler(config)
            sampler.sample()
        }
    }

    fun sampleInteractive() {
        println("대화형 텍스트 생성")
        println("종료하려면 'quit'를 입력하세요.")

        while (true) {
            print("\n프롬프트 입력: ")
            val prompt = readLine() ?: break

            if (prompt.lowercase() == "quit") break

            val config = SampleConfig(
                start = prompt,
                numSamples = 1,
                maxNewTokens = 100,
                temperature = 0.8,
                topK = 50
            )

            val sampler = Sampler(config)
            sampler.sample()
        }
    }

    // 메인 함수
    fun main() {
        // 기본 샘플링
        println("=== 기본 Shakespeare 샘플링 ===")
        sampleShakespeare()

        // 다양한 온도로 샘플링
        println("\n=== 온도 변화 실험 ===")
        sampleWithDifferentTemperatures()

        // 대화형 모드
        // sampleInteractive()
    }
}