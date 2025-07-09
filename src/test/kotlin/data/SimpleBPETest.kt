package data

import kotlinx.coroutines.runBlocking
import java.io.File

fun main() {
    println("=== SimpleBPE TinyStories 테스트 ===")

    // 테스트용 작은 샘플 데이터 생성
    val sampleFile = File("data/stories.txt")

    if (!sampleFile.exists()) {
        println("파일을 찾을 수 없습니다: ${sampleFile.absolutePath}")
        return
    }

    // 작은 샘플만 읽어서 테스트 (처음 50KB)
    val text = sampleFile.readText() //.take(5000000)
    println("샘플 텍스트 길이: ${text.length} 문자")
    println("샘플 텍스트 시작:")
    println(text.take(200) + "...")
    println()

    // SimpleBPE 초기화 및 학습
    val bpe = SimpleBPE(maxVocabSize = 100)

    println("BPE 학습 시작...")
    runBlocking {
        bpe.train(text)
    }

    println("\n=== 학습 결과 ===")
    println("최종 어휘 크기: ${bpe.getVocabSize()}")

    // 어휘 샘플 출력 (처음 50개)
    val vocab = bpe.getStoi()
    println("\n어휘 샘플 (처음 50개):")
    vocab.entries.take(50).forEach { (token, id) ->
        val displayToken = if (token.length == 1) {
            when (token) {
                "\n" -> "\\n"
                "\t" -> "\\t"
                " " -> "SPACE"
                else -> token
            }
        } else {
            token
        }
        println("$id: '$displayToken'")
    }

    // 인코딩 테스트
    val testSentences = listOf(
        "Once upon a time",
        "The little boy",
        "They played together",
        "It was a beautiful day"
    )

    println("\n=== 인코딩 테스트 ===")
    testSentences.forEach { sentence ->
        val encoded = bpe.encode(sentence)
        val compression = sentence.length.toDouble() / encoded.size
        println("원문: '$sentence'")
        println("토큰 수: ${encoded.size} (압축률: ${"%.2f".format(compression)}:1)")
        println("인코딩: $encoded")
        println()
    }

    // 통계 정보
    println("=== 통계 정보 ===")
    val totalChars = text.length
    val encoded = bpe.encode(text)
    val totalTokens = encoded.size
    val compressionRatio = totalChars.toDouble() / totalTokens

    println("원본 문자 수: $totalChars")
    println("인코딩 후 토큰 수: $totalTokens")
    println("압축률: ${"%.2f".format(compressionRatio)}:1")

    // 토큰 길이 분포
    val tokenLengths = vocab.keys.groupingBy { it.length }.eachCount()
    println("\n토큰 길이 분포:")
    tokenLengths.toSortedMap().forEach { (length, count) ->
        println("길이 $length: $count 개")
    }

    // 가장 긴 토큰들 출력
    val longestTokens = vocab.keys.filter { it.length > 5 }.sortedByDescending { it.length }.take(10)
    if (longestTokens.isNotEmpty()) {
        println("\n가장 긴 토큰들:")
        longestTokens.forEach { token ->
            println("'$token' (길이: ${token.length})")
        }
    }
}