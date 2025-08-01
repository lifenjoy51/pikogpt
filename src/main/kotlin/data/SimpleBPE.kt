package data

/**
 * String 기반 단순 BPE (Byte Pair Encoding) 구현.
 * 디버깅의 편의를 위해 Byte가 아닌 String 단위로 처리.
 */
class SimpleBPE(
    private val maxVocabSize: Int,
    private val specialTokens: List<String> = listOf("<|eos|>", UNKNOWN_TOKEN)
) {
    /**
     * 토큰과 ID 매핑을 위한 맵
     */
    private val tokenToId = mutableMapOf<String, Int>()

    /**
     * 병합 규칙을 저장하기 위한 리스트
     */
    private val merges = mutableListOf<TokenPair>()

    fun train(text: String) {
        println("BPE 학습 시작 (목표 어휘 크기: $maxVocabSize, 텍스트 길이: ${text.length})")
        val startTime = System.currentTimeMillis()

        // 예약된 특수 토큰들을 어휘에 추가. 공백을 추가하는 전처리 포함.
        var processedText = text
        specialTokens.forEachIndexed { index, token ->
            tokenToId[token] = index
            processedText = processedText.replace(token, " $token ")
        }
        println("예약된 특수 토큰: $specialTokens")

        // 모든 문자를 vocab에 추가
        val uniqueChars = processedText.toSet()
            .sorted()
        uniqueChars.forEach { char ->
            val charStr = char.toString()
            if (charStr !in tokenToId) {
                tokenToId[charStr] = tokenToId.size
            }
        }
        println("총 고유 문자 수: ${uniqueChars.size}")

        // 토큰화
        var tokens = tokenize(processedText)
        println("초기 토큰 수: ${tokens.size}")

        // 목표 어휘수 까지 병합
        var iteration = 0
        while (tokenToId.size < maxVocabSize) {
            val pairs = getPairs(tokens)
            if (pairs.isEmpty()) {
                println("더 이상 병합할 쌍이 없습니다. (반복: $iteration, 현재 어휘 크기: ${tokenToId.size})")
                break
            }

            // 가장 빈번한 쌍 찾기 (개선: 빈도와 길이를 고려한 우선순위)
            val mostCommon = selectBestPair(pairs)
            val newToken = mostCommon.toMergedToken()

            // 새 토큰을 어휘에 추가
            tokenToId[newToken] = tokenToId.size

            // 병합 규칙 저장
            merges.add(mostCommon)

            // 토큰 병합
            tokens = mergeTokens(tokens, mostCommon)

            // 로깅
            if (iteration++ % 100 == 0) {
                loggingMiddle(startTime, iteration, pairs, tokens)
            }
        }

        // 최종 로깅
        loggingEnd(startTime, processedText, tokens)
        
        // 디버그: merges 개수 확인
        println("DEBUG: 학습 완료 후 merges 개수: ${merges.size}")
    }

    /**
     * 주어진 텍스트를 토큰 ID 리스트로 인코딩
     */
    fun encode(text: String): List<Int> {
        if (text.isEmpty()) return emptyList()
        
        // 특수 토큰을 고려한 전처리 (train과 동일한 방식)
        var processedText = text
        specialTokens.forEach { token ->
            processedText = processedText.replace(token, " $token ")
        }

        // 토큰화
        var tokens: List<String> = tokenize(processedText)

        // 학습된 병합 규칙을 순서대로 적용
        println("DEBUG: encode 시작 - merges 개수: ${merges.size}, 초기 토큰 수: ${tokens.size}")
        for ((index, mergeRule) in merges.withIndex()) {
            if (tokens.size < 2) break // 더 이상 병합할 수 없음
            val beforeSize = tokens.size
            tokens = mergeTokens(tokens, mergeRule)
            if (index % 100 == 0) {
                println("DEBUG: 병합 진행 $index/${merges.size} - 토큰 수: $beforeSize -> ${tokens.size}")
            }
        }
        println("DEBUG: encode 병합 완료 - 최종 토큰 수: ${tokens.size}")
        
        // 토큰을 인덱스로 변환 (null 체크 최적화)
        val unknownTokenId = tokenToId[UNKNOWN_TOKEN] ?: 1
        var unknownCount = 0 
        val result = tokens.map { token ->
            tokenToId[token] ?: run {
                unknownCount++
                if (unknownCount <= 20) { // 첫 20개만 로깅
                    println("Unknown token found: '$token' -> using ID $unknownTokenId")
                }
                unknownTokenId
            }
        }
        
        if (unknownCount > 0) {
            println("Total unknown tokens found: $unknownCount out of ${tokens.size} tokens (${String.format("%.2f", unknownCount.toDouble() / tokens.size * 100)}%)")
        }
        
        return result
    }

    fun getVocabSize(): Int = tokenToId.size

    fun getStoi(): Map<String, Int> = tokenToId.toMap()

    fun getItos(): Map<Int, String> {
        return tokenToId.entries.associate { (token, id) -> id to token }
    }

    /**
     * 주어진 텍스트를 토큰화합니다.
     * 특수 토큰을 고려하여 청크 단위로 처리합니다.
     */
    private fun tokenize(text: String): List<String> {
        println("토큰화 시작 (텍스트 길이: ${text.length})")
        val tokens = mutableListOf<String>()
        val sortedSpecialTokens = specialTokens.sortedByDescending { it.length }
        var i = 0

        while (i < text.length) {
            var found = false

            // 특수 토큰 체크 (가장 긴 것부터)
            for (token in sortedSpecialTokens) {
                if (i + token.length <= text.length &&
                    text.startsWith(token, i)) {
                    tokens.add(token)
                    i += token.length
                    found = true
                    break
                }
            }

            if (!found) {
                val charStr = text[i].toString()
                tokens.add(charStr)  // train에서 이미 모든 문자를 vocabulary에 추가했으므로 무조건 추가
                i++
            }
        }

        return tokens
    }

    /**
     * 연속된 토큰 쌍의 빈도를 계산
     */
    private fun getPairs(tokens: List<String>): Map<TokenPair, Int> {
        val totalTokens = tokens.size
        val estimatedPairs = maxOf(totalTokens / 2, 1000)
        val pairs = HashMap<TokenPair, Int>(estimatedPairs)

        for (i in 0..<(totalTokens - 1)) {
            val first = tokens[i]
            val second = tokens[i + 1]

            if (first in specialTokens || second in specialTokens) {
                continue
            }

            val pair = TokenPair(first, second)
            val mergedToken = pair.toMergedToken()

            // 병합된 토큰의 공백 개수 확인 (최대 1개까지만 허용)
            val spaceCount = mergedToken.count { it == ' ' }
            if (spaceCount <= 1) {
                pairs.merge(pair, 1, Int::plus)
            }
        }

        return pairs
    }

    /**
     * 토큰 리스트에서 병합 규칙을 적용하여 병합된 토큰 리스트를 반환
     * O(n) 시간복잡도로 최적화
     */
    private fun mergeTokens(tokens: List<String>, pair: TokenPair): List<String> {
        val newTokens = mutableListOf<String>()
        val specialTokenSet = specialTokens.toSet()
        var i = 0

        while (i < tokens.size) {
            if (i < tokens.size - 1 && tokens[i] == pair.first && tokens[i + 1] == pair.second) {
                // 특수 토큰이 포함된 쌍은 병합하지 않음
                if (tokens[i] in specialTokenSet || tokens[i + 1] in specialTokenSet) {
                    newTokens.add(tokens[i])
                    i++
                } else {
                    newTokens.add(pair.toMergedToken())
                    i += 2  // 두 토큰을 건너뛰기
                }
            } else {
                newTokens.add(tokens[i])
                i++
            }
        }

        return newTokens
    }

    /**
     * 중간 진행 상황 로깅
     */
    private fun loggingMiddle(
        startTime: Long,
        iteration: Int,
        pairs: Map<TokenPair, Int>,
        tokens: List<String>
    ) {
        val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
        val progress = (tokenToId.size.toDouble() / maxVocabSize * 100).toInt()
        println("\n=== 진행 상황 리포트 ===\n${iteration}번 병합 완료")
        println("\n반복 $iteration: 총 쌍 ${pairs.size}개")
        pairs.toList()
            .sortedByDescending { it.second }
            .take(5)
            .forEach { (pair, count) ->
                println("  (${pair.first} + ${pair.second}) -> '${count}회'")
            }
        println("어휘 크기: ${tokenToId.size}/$maxVocabSize (${progress}%)")
        println("현재 토큰 수: ${tokens.size}")
        println("소요 시간: ${elapsed}s")
        println("평균 병합 속도: ${String.format("%.2f", iteration / elapsed)} 병합/초")
    }

    /**
     * 최종 통계 출력
     */
    private fun loggingEnd(
        startTime: Long,
        processedText: String,
        tokens: List<String>
    ) {
        val totalTime = (System.currentTimeMillis() - startTime) / 1000.0
        val actualMerges = merges.size
        val initialTokenCount = processedText.length  // 대략적인 초기 문자 수
        val compressionRatio = String.format("%.2f", tokens.size.toDouble() / initialTokenCount * 100)

        println("\n=== BPE 학습 완료 ===")
        println("최종 어휘 크기: ${tokenToId.size}/$maxVocabSize")
        println("실제 병합 횟수: $actualMerges")
        println("최종 토큰 수: ${tokens.size}")
        println("압축률: $compressionRatio% (원본 문자 수 대비)")
        println("총 소요 시간: ${totalTime}s")
        println("평균 병합 속도: ${String.format("%.2f", actualMerges / totalTime)} 병합/초")

        // 최종 어휘 분석
        println("\n최종 어휘 구성:")
        println("  특수 토큰: ${specialTokens.size}개")
        println("  병합된 토큰: ${actualMerges}개")

        // 가장 긴 토큰들 출력
        val longTokens = tokenToId.keys.filter { it !in specialTokens }
            .sortedByDescending { it.length }.take(10)
        println("\n가장 긴 토큰 상위 10개:")
        longTokens.forEach { token ->
            println("  '$token' (길이: ${token.length})")
        }
    }

    /**
     * 최적의 병합 쌍을 선택합니다.
     * 빈도뿐만 아니라 토큰 길이와 의미적 가치를 고려합니다.
     */
    private fun selectBestPair(pairs: Map<TokenPair, Int>): TokenPair {
        return pairs.maxByOrNull { (pair, frequency) ->
            // 기본 점수: 빈도
            var score = frequency.toDouble()
            
            // 길이 보너스: 더 긴 토큰을 선호 (하지만 너무 길지 않게)
            val totalLength = pair.first.length + pair.second.length
            if (totalLength <= 6) { // 적당한 길이의 토큰 선호
                score *= (1.0 + totalLength * 0.1)
            }
            
            // 공백 패널티: 공백이 많은 토큰은 선호하지 않음
            val spaceCount = pair.first.count { it == ' ' } + pair.second.count { it == ' ' }
            if (spaceCount > 0) {
                score *= (1.0 - spaceCount * 0.2)
            }
            
            score
        }?.key ?: pairs.keys.first()
    }

    /**
     * 두 토큰의 쌍을 나타내는 데이터 클래스
     */
    data class TokenPair(
        val first: String,
        val second: String
    ) {
        fun toMergedToken(): String = "$first$second"
    }

    companion object {
        const val UNKNOWN_TOKEN = "<|unk|>" // 알 수 없는 토큰, BPE에서 자주 사용되는 특수 토큰
    }

}