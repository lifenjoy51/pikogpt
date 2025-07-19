package sample

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext

fun main() {
    runBlocking {
        test()
        //testVariousPrompts()
    }
}

suspend fun testVariousPrompts() = withContext(Dispatchers.Default) {
    val prompts = listOf(
        "once upon a time",
        "the cat and the dog",
        "the boy saw a",
        "the girl said,",
        "they lived happily"
    )

    val loss = listOf("16", "17", "18", "19", "20", "21", "22", "23")

    // 모든 loss 값을 병렬로 처리
    val result = loss.map { l ->
        async {
            val config = SampleConfig(
                modelDirectoryPath = "model/71200/$l",
                numberOfSamples = 5,
                maximumNewTokens = 50,
                topKFilteringSize = 10
            )
            val sampler = Sampler(config)

            // 각 loss 값에 대해 모든 프롬프트를 병렬로 처리
            prompts.map { prompt ->
                async {
                    //println("\n=== Loss: $l, Prompt: $prompt ===")
                    sampler.sample(prompt)
                }
            }.awaitAll()
        }
    }.awaitAll().flatten()

    println(result.size)

    result.forEach {
        println("\n=== ModelId: ${it.uid}, Prompt: ${it.prompt} ===")
        it.results.forEach { line ->
            println(line)
        }
    }
}


suspend fun test() {
    val config = SampleConfig(
        modelInitializationMode = "resume",
        modelDirectoryPath = "model/71200/20",
        numberOfSamples = 3,
    )

    val sampler = Sampler(config)
    val result = sampler.sample("the cat and the dog")
    println(result)
}

fun sampleInteractive() {
    println("대화형 텍스트 생성")
    println("종료하려면 'quit'를 입력하세요.")

    val config = SampleConfig(
        numberOfSamples = 1,
        maximumNewTokens = 100,
        samplingTemperature = 0.8f,
        topKFilteringSize = 50
    )

    val sampler = Sampler(config)

    runBlocking {
        while (true) {
            print("\n프롬프트 입력: ")
            val prompt = readlnOrNull() ?: break

            if (prompt.lowercase() == "quit") break

            sampler.sample(prompt)
        }
    }
}
