package data


/**
 * SimpleBPE를 사용한 Shakespeare 데이터셋 토크나이저
 */
class SimpleBPE( val numMerges: Int = 1000) {
    private val vocab = mutableMapOf<String, Int>()
    private val merges = mutableListOf<Pair<Pair<Int, Int>, Int>>()
    private var vocabSize = 256 // 기본 바이트 값
    
    init {
        // 기본 바이트 토큰 초기화 (0-255)
        for (i in 0 until 256) {
            vocab[i.toChar().toString()] = i
        }
    }
    
    fun train(text: String) {
        println("BPE 학습 시작 (병합 횟수: $numMerges)")
        var tokens = text.toByteArray().map { it.toInt() and 0xFF }
        
        repeat(numMerges) { iteration ->
            val pairs = getPairs(tokens)
            if (pairs.isEmpty()) {
                println("더 이상 병합할 쌍이 없습니다. (반복: $iteration)")
                return
            }
            
            val mostCommon = pairs.entries.maxByOrNull { it.value }?.key ?: return
            val newToken = vocabSize++
            
            // 병합 규칙 저장
            merges.add(Pair(mostCommon, newToken))
            vocab["${mostCommon.first},${mostCommon.second}"] = newToken
            
            // 토큰 병합
            tokens = merge(tokens, mostCommon, newToken)
            
            if ((iteration + 1) % 100 == 0) {
                println("진행 상황: ${iteration + 1}/$numMerges 병합 완료")
            }
        }
        
        println("BPE 학습 완료! 최종 어휘 크기: $vocabSize")
    }
    
    private fun getPairs(tokens: List<Int>): Map<Pair<Int, Int>, Int> {
        val pairs = mutableMapOf<Pair<Int, Int>, Int>()
        for (i in 0 until tokens.size - 1) {
            val pair = Pair(tokens[i], tokens[i + 1])
            pairs[pair] = pairs.getOrDefault(pair, 0) + 1
        }
        return pairs
    }
    
    private fun merge(tokens: List<Int>, pair: Pair<Int, Int>, newToken: Int): List<Int> {
        val result = mutableListOf<Int>()
        var i = 0
        while (i < tokens.size) {
            if (i < tokens.size - 1 && tokens[i] == pair.first && tokens[i + 1] == pair.second) {
                result.add(newToken)
                i += 2
            } else {
                result.add(tokens[i])
                i += 1
            }
        }
        return result
    }
    
    fun encode(text: String): List<Int> {
        // 먼저 바이트로 변환
        var tokens = text.toByteArray().map { it.toInt() and 0xFF }
        
        // 학습된 병합 규칙을 순서대로 적용
        for ((pair, newToken) in merges) {
            tokens = merge(tokens, pair, newToken)
        }
        
        return tokens
    }
    
    fun getVocabSize(): Int = vocabSize
}
