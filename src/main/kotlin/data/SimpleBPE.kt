package data

import kotlinx.coroutines.*
import kotlin.math.min
import java.util.concurrent.ConcurrentHashMap


/**
 * SimpleBPE를 사용한 Shakespeare 데이터셋 토크나이저 (String 기반, 최적화)
 * 예약된 특수 토큰 지원
 */
class SimpleBPE(
    val maxVocabSize: Int = 100,
    private val specialTokens: List<String> = listOf("<|eos|>", "<|unk|>")
) {
    private val vocab = mutableMapOf<String, Int>()
    private val merges = mutableListOf<Pair<Pair<String, String>, String>>()
    private var vocabSize = 0
    
    // 직접적인 쌍 빈도 계산 (캐시 제거로 단순화)
    private var lastPairResult: Map<Pair<String, String>, Int>? = null
    
    suspend fun train(text: String) = coroutineScope {
        println("BPE 학습 시작 (목표 어휘 크기: $maxVocabSize, 텍스트 길이: ${text.length})")
        val startTime = System.currentTimeMillis()
        
        // 1. 먼저 예약된 특수 토큰들을 어휘에 추가
        specialTokens.forEachIndexed { index, token ->
            vocab[token] = index
        }
        vocabSize = specialTokens.size
        println("예약된 특수 토큰: $specialTokens")
        
        // 2. 텍스트 전처리 (간단하고 효율적으로)
        var processedText = text
        specialTokens.forEach { token ->
            processedText = processedText.replace(token, " $token ")
        }
        
        // 3. 문자 빈도 계산 (병렬 처리)
        println("\n=== 문자 빈도 분석 시작 ===")
        val charFrequency = calculateCharFrequencyParallel(processedText)
        
        println("총 고유 문자 수: ${charFrequency.size}")
        val sortedByFreq = charFrequency.toList().sortedByDescending { it.second }
        println("가장 빈번한 문자 상위 10개:")
        sortedByFreq.take(10).forEach { (char, count) ->
            val displayChar = if (char == '\n') "\\n" else char.toString()
            println("  '$displayChar': ${count}회")
        }
        
        // 4. 모든 문자를 vocab에 추가
        val validChars = charFrequency.keys.sorted()
        
        validChars.forEach { char ->
            if (char.toString() !in vocab) {
                vocab[char.toString()] = vocabSize++
            }
        }
        println("\n고유 문자 수: ${validChars.size}, 총 기본 어휘 크기: $vocabSize")

        // 고유 단어 수 계산 생략 (메모리 절약)
        // val uniqueWords = calculateUniqueWordsParallel(processedText)
        println("메모리 절약을 위해 고유 단어 수 계산 생략")

        // 메모리 효율적인 토큰화
        println("\n=== 초기 토큰화 시작 ===")
        val tokens = tokenizeMemoryEfficient(processedText)
        println("초기 토큰 수: ${tokens.size}")
        
        // 토큰 분포 분석 생략 (메모리 절약)
        println("고유 토큰 수: ${vocab.size} (문자 수준)")
        println("메모리 절약을 위해 토큰 분포 분석 생략")
        
        println("\n=== BPE 병합 과정 시작 ===")
        var iteration = 0
        while (vocabSize < maxVocabSize) {
            val pairs = getPairsParallel(tokens)
            if (pairs.isEmpty()) {
                println("더 이상 병합할 쌍이 없습니다. (반복: $iteration, 현재 어휘 크기: $vocabSize)")
                break
            }
            
            if (iteration < 10 || iteration % 10 == 0) {
                println("\n반복 $iteration: 총 쌍 ${pairs.size}개")
                if (pairs.isNotEmpty()) {
                    println("빈도 상위 10개 쌍:")
                    pairs.toList().sortedByDescending { it.second }.take(10).forEach { (pair, count) ->
                        println("  (${pair.first} + ${pair.second}) -> '${count}회'")
                    }
                }
            }
            
            // 가장 빈번한 쌍 찾기
            var mostCommon: Pair<String, String>? = null
            var maxCount = 0
            for ((pair, count) in pairs) {
                if (count > maxCount) {
                    maxCount = count
                    mostCommon = pair
                }
            }
            
            mostCommon ?: break
            val newToken = "${mostCommon.first}${mostCommon.second}"
            
            // 토큰에 포함된 공백 개수 확인 (최대 1개까지만 허용)
            val spaceCount = newToken.count { it == ' ' }
            if (spaceCount > 1) {
                if (iteration < 10 || iteration % 50 == 0) {
                    println("병합 건너뜀: '$newToken' (공백 ${spaceCount}개로 제한 초과)")
                }
                // 현재 쌍을 제외하고 다음으로 빈번한 쌍 찾기
                val filteredPairs = pairs.filter { (pair, _) ->
                    val testToken = "${pair.first}${pair.second}"
                    testToken.count { it == ' ' } <= 1
                }
                if (filteredPairs.isEmpty()) {
                    println("공백 제한 조건을 만족하는 쌍이 없습니다.")
                    break
                }
                
                // 가장 빈번한 유효한 쌍 다시 찾기
                var validMostCommon: Pair<String, String>? = null
                var validMaxCount = 0
                for ((pair, count) in filteredPairs) {
                    if (count > validMaxCount) {
                        validMaxCount = count
                        validMostCommon = pair
                    }
                }
                
                validMostCommon ?: break
                val validNewToken = "${validMostCommon.first}${validMostCommon.second}"
                
                val tokensBefore = tokens.size
                
                // 새 토큰을 어휘에 추가
                vocab[validNewToken] = vocabSize++
                
                // 병합 규칙 저장
                merges.add(Pair(validMostCommon, validNewToken))
                
                // 토큰 병합 (in-place 수정으로 메모리 절약)
                mergeInPlace(tokens, validMostCommon, validNewToken)
                
                val tokensAfter = tokens.size
                val tokenReduction = tokensBefore - tokensAfter
                
                if (iteration < 10 || iteration % 50 == 0) {
                    println("토큰 수 변화: $tokensBefore -> $tokensAfter (-$tokenReduction)")
                }
                
                iteration++
                if (iteration % 100 == 0) {
                    val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
                    val progress = (vocabSize.toDouble() / maxVocabSize * 100).toInt()
                    println("\n=== 진행 상황 리포트 ===\n${iteration}번 병합 완료")
                    println("어휘 크기: $vocabSize/$maxVocabSize (${progress}%)")
                    println("현재 토큰 수: ${tokens.size}")
                    println("소요 시간: ${elapsed}s")
                    println("평균 병합 속도: ${String.format("%.2f", iteration / elapsed)} 병합/초")
                }
                continue
            }
            
            val tokensBefore = tokens.size
            
            // 새 토큰을 어휘에 추가
            vocab[newToken] = vocabSize++
            
            // 병합 규칙 저장
            merges.add(Pair(mostCommon, newToken))
            
            // 토큰 병합 (in-place 수정으로 메모리 절약)
            mergeInPlace(tokens, mostCommon, newToken)
            
            val tokensAfter = tokens.size
            val tokenReduction = tokensBefore - tokensAfter
            
            if (iteration < 10 || iteration % 50 == 0) {
                println("토큰 수 변화: $tokensBefore -> $tokensAfter (-$tokenReduction)")
            }
            
            iteration++
            if (iteration % 100 == 0) {
                val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
                val progress = (vocabSize.toDouble() / maxVocabSize * 100).toInt()
                println("\n=== 진행 상황 리포트 ===\n${iteration}번 병합 완료")
                println("어휘 크기: $vocabSize/$maxVocabSize (${progress}%)")
                println("현재 토큰 수: ${tokens.size}")
                println("소요 시간: ${elapsed}s")
                println("평균 병합 속도: ${String.format("%.2f", iteration / elapsed)} 병합/초")
            }
        }
        
        val totalTime = (System.currentTimeMillis() - startTime) / 1000.0
        val actualMerges = merges.size
        val initialTokenCount = processedText.length  // 대략적인 초기 문자 수
        val compressionRatio = String.format("%.2f", tokens.size.toDouble() / initialTokenCount * 100)
        
        println("\n=== BPE 학습 완료 ===")
        println("최종 어휘 크기: $vocabSize/$maxVocabSize")
        println("실제 병합 횟수: $actualMerges")
        println("최종 토큰 수: ${tokens.size}")
        println("압축률: $compressionRatio% (원본 문자 수 대비)")
        println("총 소요 시간: ${totalTime}s")
        println("평균 병합 속도: ${String.format("%.2f", actualMerges / totalTime)} 병합/초")
        
        // 최종 어휘 분석
        println("\n최종 어휘 구성:")
        println("  특수 토큰: ${specialTokens.size}개")
        println("  단일 문자: ${validChars.size}개")
        println("  병합된 토큰: ${actualMerges}개")
        
        // 가장 긴 토큰들 출력
        val longTokens = vocab.keys.filter { it !in specialTokens }
            .sortedByDescending { it.length }.take(10)
        println("\n가장 긴 토큰 상위 10개:")
        longTokens.forEach { token ->
            println("  '$token' (길이: ${token.length})")
        }
        
        // 캐시 정리
        lastPairResult = null
    }
    
    private fun getPairsOptimized(tokens: List<String>): Map<Pair<String, String>, Int> {
        val pairs = HashMap<Pair<String, String>, Int>()  // HashMap 사용
        val size = tokens.size
        
        // 특수 토큰 체크를 Set으로 최적화
        val specialTokenSet = specialTokens.toSet()
        
        for (i in 0 until size - 1) {
            val first = tokens[i]
            val second = tokens[i + 1]
            
            // 특수 토큰이 포함된 쌍은 병합하지 않음
            if (first in specialTokenSet || second in specialTokenSet) {
                continue
            }
            
            val pair = Pair(first, second)
            pairs[pair] = (pairs[pair] ?: 0) + 1
        }
        
        return pairs
    }
    
    private fun mergeInPlace(tokens: MutableList<String>, pair: Pair<String, String>, newToken: String) {
        // 새로운 리스트 생성으로 O(n) 시간복잡도로 최적화
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
                    newTokens.add(newToken)
                    i += 2  // 두 토큰을 건너뛰기
                }
            } else {
                newTokens.add(tokens[i])
                i++
            }
        }
        tokens.clear()
        tokens.addAll(newTokens)
    }
    
    private fun tokenizeWithSpecialTokens(text: String): List<String> {
        val tokens = mutableListOf<String>()
        var i = 0
        
        while (i < text.length) {
            var found = false
            
            // 특수 토큰 체크 (가장 긴 것부터)
            for (token in specialTokens.sortedByDescending { it.length }) {
                if (i + token.length <= text.length && 
                    text.substring(i, i + token.length) == token) {
                    tokens.add(token)
                    i += token.length
                    found = true
                    break
                }
            }
            
            if (!found) {
                val charStr = text[i].toString()
                if (charStr in vocab) {
                    tokens.add(charStr)
                } else {
                    tokens.add("<|unk|>")
                }
                i++
            }
        }
        
        return tokens
    }
    
    /**
     * 메모리 효율적인 토큰화
     */
    private fun tokenizeMemoryEfficient(text: String): MutableList<String> {
        println("메모리 효율적인 토큰화 시작 (텍스트 길이: ${text.length})")
        
        // 큰 텍스트의 경우 청크 단위로 처리
        if (text.length > 1_000_000) {
            val tokens = mutableListOf<String>()
            val chunkSize = 100_000
            var processed = 0
            
            var start = 0
            while (start < text.length) {
                val end = min(start + chunkSize, text.length)
                val chunk = text.substring(start, end)
                
                // 청크도 특수 토큰을 고려해서 처리
                tokens.addAll(tokenizeWithSpecialTokens(chunk))
                
                processed += chunk.length
                val progress = (processed.toDouble() / text.length * 100).toInt()
                println("토큰화 진행률: ${progress}% (${processed}/${text.length})")
                
                start = end
            }
            
            return tokens
        } else {
            // 작은 텍스트는 기존 방식 사용
            return tokenizeWithSpecialTokens(text).toMutableList()
        }
    }
    
    fun encode(text: String): List<Int> {
        if (text.isEmpty()) return emptyList()
        
        // 특수 토큰을 고려한 전처리
        var processedText = text
        specialTokens.forEach { token ->
            processedText = processedText.replace(token, " $token ")
        }
        
        // 토큰화 (재사용 가능한 리스트)
        val tokens = tokenizeWithSpecialTokens(processedText).toMutableList()
        
        // 학습된 병합 규칙을 순서대로 적용 (in-place 수정)
        for ((pair, newToken) in merges) {
            if (tokens.size < 2) break // 더 이상 병합할 수 없음
            mergeInPlace(tokens, pair, newToken)
        }
        
        // 토큰을 인덱스로 변환 (null 체크 최적화)
        return tokens.map { vocab[it] ?: run {
            // 알 수 없는 토큰은 <|unk|> 토큰으로 대체
            vocab["<|unk|>"] ?: 1
        } }
    }
    
    fun getVocabSize(): Int = vocabSize

    fun getStoi(): Map<String, Int> = vocab.toMap()

    fun getItos(): Map<Int, String> {
        return vocab.entries.associate { (token, id) -> id to token }
    }

    // 병렬 처리 메서드들
    
    /**
     * 문자 빈도를 병렬로 계산
     */
    private suspend fun calculateCharFrequencyParallel(text: String): Map<Char, Int> = 
        withContext(Dispatchers.Default) {
            val numCores = Runtime.getRuntime().availableProcessors()
            val chunkSize = maxOf(1000, text.length / numCores)
            
            if (text.length < chunkSize * 2) {
                // 작은 텍스트는 순차 처리
                val frequency = mutableMapOf<Char, Int>()
                text.forEach { char ->
                    if (char.toString() !in specialTokens && char != '\n') {
                        frequency[char] = (frequency[char] ?: 0) + 1
                    }
                }
                return@withContext frequency
            }
            
            // 큰 텍스트는 병렬 처리
            val chunks = mutableListOf<String>()
            var start = 0
            while (start < text.length) {
                val end = min(start + chunkSize, text.length)
                chunks.add(text.substring(start, end))
                start = end
            }
            
            chunks.map { chunk ->
                async {
                    val frequency = mutableMapOf<Char, Int>()
                    chunk.forEach { char ->
                        if (char.toString() !in specialTokens && char != '\n') {
                            frequency[char] = (frequency[char] ?: 0) + 1
                        }
                    }
                    frequency
                }
            }.awaitAll().fold(mutableMapOf<Char, Int>()) { acc, freq ->
                freq.forEach { (char, count) ->
                    acc[char] = (acc[char] ?: 0) + count
                }
                acc
            }
        }
    
    /**
     * 고유 단어를 병렬로 계산
     */
    private suspend fun calculateUniqueWordsParallel(text: String): Set<String> = 
        withContext(Dispatchers.Default) {
            val numCores = Runtime.getRuntime().availableProcessors()
            val chunkSize = maxOf(1000, text.length / numCores)
            
            if (text.length < chunkSize * 2) {
                // 작은 텍스트는 순차 처리
                val uniqueWords = mutableSetOf<String>()
                val wordRegex = Regex("\\w+")
                wordRegex.findAll(text.lowercase()).forEach { match ->
                    uniqueWords.add(match.value)
                }
                return@withContext uniqueWords
            }
            
            // 큰 텍스트는 병렬 처리
            val chunks = mutableListOf<String>()
            var start = 0
            while (start < text.length) {
                val end = min(start + chunkSize, text.length)
                chunks.add(text.substring(start, end))
                start = end
            }
            
            val wordRegex = Regex("\\w+")
            chunks.map { chunk ->
                async {
                    val words = mutableSetOf<String>()
                    wordRegex.findAll(chunk.lowercase()).forEach { match ->
                        words.add(match.value)
                    }
                    words
                }
            }.awaitAll().fold(mutableSetOf<String>()) { acc, words ->
                acc.apply { addAll(words) }
            }
        }
    
    /**
     * 메모리 효율적인 토큰 쌍 계산 (스트리밍 방식)
     */
    private suspend fun getPairsParallel(tokens: List<String>): Map<Pair<String, String>, Int> = 
        withContext(Dispatchers.Default) {
            val totalTokens = tokens.size
            //println("토큰 쌍 계산 시작 (토큰 수: ${totalTokens})")
            
            // 메모리 사용량이 많을 때는 순차 처리로 전환
            if (totalTokens > 1_000_000) {
                //println("대용량 데이터 감지 - 메모리 효율적인 순차 처리 사용")
                return@withContext getPairsOptimizedMemoryEfficient(tokens)
            }
            
            // 중간 크기 데이터는 제한된 병렬 처리
            val numCores = min(6, Runtime.getRuntime().availableProcessors()) // 최대 4개 코어만 사용
            val chunkSize = maxOf(50_000, totalTokens / numCores) // 더 큰 청크 사용
            
            if (totalTokens < chunkSize) {
                return@withContext getPairsOptimized(tokens)
            }
            
            println("제한된 병렬 처리 사용 (코어: ${numCores}, 청크 크기: ${chunkSize})")
            
            val specialTokenSet = specialTokens.toSet()
            val chunks = mutableListOf<IntRange>()
            var start = 0
            
            while (start < totalTokens) {
                val end = min(start + chunkSize, totalTokens)
                chunks.add(start until end)
                start = end
            }
            
            val finalPairs = ConcurrentHashMap<Pair<String, String>, Int>()
            
            chunks.chunked(2).forEach { chunkBatch -> // 배치로 처리해서 메모리 압박 줄임
                chunkBatch.map { range ->
                    async {
                        val pairs = HashMap<Pair<String, String>, Int>(1000) // 초기 크기 제한
                        
                        for (i in range.first until min(range.last, totalTokens - 1)) {
                            val first = tokens[i]
                            val second = tokens[i + 1]
                            
                            if (first in specialTokenSet || second in specialTokenSet) {
                                continue
                            }
                            
                            val pair = Pair(first, second)
                            pairs[pair] = (pairs[pair] ?: 0) + 1
                        }
                        
                        // 결과를 메인 맵에 병합
                        pairs.forEach { (pair, count) ->
                            finalPairs.merge(pair, count, Int::plus)
                        }
                    }
                }.awaitAll()
                
                // 메모리 정리를 위한 작은 지연
                if (chunks.size > 4) {
                    yield()
                }
            }
            
            finalPairs
        }
    
    /**
     * 메모리 효율적인 순차 쌍 계산
     */
    private fun getPairsOptimizedMemoryEfficient(tokens: List<String>): Map<Pair<String, String>, Int> {
        val pairs = HashMap<Pair<String, String>, Int>(10000) // 적당한 초기 크기
        val specialTokenSet = specialTokens.toSet()
        val totalTokens = tokens.size
        
        for (i in 0 until totalTokens - 1) {
            val first = tokens[i]
            val second = tokens[i + 1]
            
            if (first in specialTokenSet || second in specialTokenSet) {
                continue
            }
            
            val pair = Pair(first, second)
            pairs[pair] = (pairs[pair] ?: 0) + 1

        }
        
        return pairs
    }
}
