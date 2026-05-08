package data

import data.CharBPE.Companion.UNKNOWN_TOKEN
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.serialization.json.*

fun main() {
    println("=== 토큰 분포 분석 ===")

    // 1. Meta 정보 로드
    val metaFile = File("/Users/joey51/works/pikogpt/data/1k/meta.json")
    val metaJson = Json.parseToJsonElement(metaFile.readText()).jsonObject
    val vocabularySize = metaJson["vocabularySize"]!!.jsonPrimitive.int
    val indexToString = metaJson["indexToString"]!!.jsonObject

    println("어휘 크기: $vocabularySize")
    println("토큰 ID 1번: ${indexToString["1"]?.jsonPrimitive?.content}")

    // 2. 바이너리 데이터 로드 함수
    fun loadBinaryData(filePath: String): IntArray {
        val file = File(filePath)
        val bytes = file.readBytes()
        val buffer = ByteBuffer.wrap(bytes)
        buffer.order(ByteOrder.BIG_ENDIAN)

        val tokenData = IntArray(bytes.size / 4)
        for (i in tokenData.indices) {
            tokenData[i] = buffer.getInt()
        }
        return tokenData
    }

    // 3. 훈련 데이터 분석
    println("\n=== 훈련 데이터 분석 ===")
    val trainData = loadBinaryData("/Users/joey51/works/pikogpt/data/1k/train.bin")
    println("훈련 데이터 토큰 개수: ${trainData.size}")

    // 토큰 분포 계산
    val trainTokenCounts = IntArray(vocabularySize) { 0 }
    for (token in trainData) {
        if (token >= 0 && token < vocabularySize) {
            trainTokenCounts[token]++
        }
    }

    println("토큰 ID 1번(전각공백 $UNKNOWN_TOKEN) 출현 횟수: ${trainTokenCounts[1]}")
    println("토큰 ID 1번 비율: ${trainTokenCounts[1].toDouble() / trainData.size * 100}%")

    // 4. 검증 데이터 분석
    println("\n=== 검증 데이터 분석 ===")
    val valData = loadBinaryData("/Users/joey51/works/pikogpt/data/1k/val.bin")
    println("검증 데이터 토큰 개수: ${valData.size}")

    val valTokenCounts = IntArray(vocabularySize) { 0 }
    for (token in valData) {
        if (token >= 0 && token < vocabularySize) {
            valTokenCounts[token]++
        }
    }

    println("토큰 ID 1번(전각공백 '$UNKNOWN_TOKEN') 출현 횟수: ${valTokenCounts[1]}")
    println("토큰 ID 1번 비율: ${valTokenCounts[1].toDouble() / valData.size * 100}%")

    // 5. 가장 빈번한 토큰들 분석
    println("\n=== 가장 빈번한 토큰 TOP 10 (훈련 데이터) ===")
    val topTokens = trainTokenCounts.withIndex()
        .sortedByDescending { it.value }

    for ((index, entry) in topTokens.withIndex()) {
        val tokenId = entry.index
        val count = entry.value
        val tokenString = indexToString[tokenId.toString()]?.jsonPrimitive?.content ?: "???"
        val percentage = count.toDouble() / trainData.size * 100
        println("${index + 1}. ID: $tokenId, 문자: '$tokenString', 횟수: $count (${percentage}%)")
    }

    // 7. 데이터 무결성 검사
    println("\n=== 데이터 무결성 검사 ===")
    val invalidTrainTokens = trainData.count { it < 0 || it >= vocabularySize }
    val invalidValTokens = valData.count { it < 0 || it >= vocabularySize }

    println("훈련 데이터 유효하지 않은 토큰: $invalidTrainTokens")
    println("검증 데이터 유효하지 않은 토큰: $invalidValTokens")

    if (invalidTrainTokens > 0) {
        println("유효하지 않은 훈련 토큰들:")
        trainData.forEachIndexed { index, token ->
            if (token < 0 || token >= vocabularySize) {
                println("  위치 $index: $token")
            }
        }
    }

    // 8. 토큰 범위 분석
    println("\n=== 토큰 범위 분석 ===")
    println("훈련 데이터 - 최소 토큰: ${trainData.minOrNull()}, 최대 토큰: ${trainData.maxOrNull()}")
    println("검증 데이터 - 최소 토큰: ${valData.minOrNull()}, 최대 토큰: ${valData.maxOrNull()}")
}
