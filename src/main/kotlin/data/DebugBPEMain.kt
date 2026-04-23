package data

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import java.io.File

fun main() {
    // 기존 BPE 모델 로드
    val bpe = SimpleBPE(maxVocabSize = 1000)

    // meta.json에서 vocabulary 정보 로드
    val metaFile = File("/Users/joey51/works/pikogpt/data/1k/meta.json")
    val metaJson = Json.parseToJsonElement(metaFile.readText()).jsonObject

    // BPE 객체의 내부 상태를 복원해야 함 (이건 불가능하므로 다른 방법 필요)

    // 간단한 텍스트로 테스트
    val testText = "the boy and the girl lived together in a small house. <|eos|>"
    println("원본 텍스트: '$testText'")

    // 특수 토큰 전처리 시뮬레이션
    var processedText = testText
    val specialTokens = listOf("<|eos|>", "<|unk|>")
    specialTokens.forEach { token ->
        processedText = processedText.replace(token, " $token ")
    }
    println("전처리된 텍스트: '$processedText'")

    // 문자 단위로 토큰화
    val tokens = mutableListOf<String>()
    var i = 0
    while (i < processedText.length) {
        var found = false
        // 특수 토큰 체크
        for (token in specialTokens.sortedByDescending { it.length }) {
            if (i + token.length <= processedText.length &&
                processedText.startsWith(token, i)) {
                tokens.add(token)
                i += token.length
                found = true
                break
            }
        }
        if (!found) {
            tokens.add(processedText[i].toString())
            i++
        }
    }

    println("초기 토큰들: $tokens")
    println("토큰 개수: ${tokens.size}")

    // vocabulary에서 찾을 수 없는 토큰들 확인
    val notFound = mutableListOf<String>()
    tokens.forEach { token ->
        if (metaJson.get("stringToIndex")?.jsonObject?.get(token) == null) {
            notFound.add(token)
        }
    }

    println("vocabulary에 없는 토큰들: $notFound")
    println("vocabulary에 없는 토큰 비율: ${notFound.size.toDouble() / tokens.size * 100}%")

    // BPE 병합 시뮬레이션 후 확인
    println("\n=== BPE 병합 없이 encode 시뮬레이션 ===")
    val finalTokens = tokens // 병합 규칙이 없으므로 그대로 유지
    val notFoundAfterMerge = mutableListOf<String>()
    finalTokens.forEach { token ->
        if (metaJson.get("stringToIndex")?.jsonObject?.get(token) == null) {
            notFoundAfterMerge.add(token)
        }
    }
    println("병합 후 vocabulary에 없는 토큰들: $notFoundAfterMerge")
    println("병합 후 vocabulary에 없는 토큰 비율: ${notFoundAfterMerge.size.toDouble() / finalTokens.size * 100}%")
}
