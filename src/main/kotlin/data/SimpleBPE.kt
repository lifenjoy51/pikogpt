package data

/**
 * 문자열 기반의 단순 BPE (Byte Pair Encoding) 구현.
 * Byte 단위가 아닌 String 단위로 처리해 디버깅이 쉽다.
 *
 * 작은 교육용 모델을 위해 다음 옵션을 지원한다:
 *
 *   - [lowercase] : 입력을 소문자로 정규화. 대/소문자 분기로 어휘가 쪼개지는 걸 방지.
 *   - [useWordPreTokenize] : 학습·인코딩 전에 공백 기반으로 "단어 조각"을 먼저 분리하고,
 *       BPE merge를 각 단어 "안에서만" 수행 (GPT-2 스타일). 단어 경계를 넘는 이상한 토큰
 *       (예: "the cat"이 한 토큰)이 생기지 않는다.
 *   - [standardBpeScoring] : 병합 시 "가장 빈번한 쌍" 기준(표준 BPE). 기본값 true.
 *       false면 길이 보너스/공백 페널티 같은 휴리스틱을 사용.
 *   - [verbose] : 학습·인코딩 중 진행 상황을 출력할지. 기본값 false.
 *
 * **학습된 상태**(어휘 + 병합 규칙)는 [getStoi] + [getMerges]로 내보낼 수 있고,
 * [restore]로 다시 주입할 수 있다 → `Sampler`가 학습 때와 **동일한 토큰화**를 재생할 수 있다.
 */
class SimpleBPE(
    private val maxVocabSize: Int,
    private val specialTokens: List<String> = listOf("<|eos|>", UNKNOWN_TOKEN, BOS_TOKEN),
    private val lowercase: Boolean = false,
    private val useWordPreTokenize: Boolean = false,
    private val standardBpeScoring: Boolean = true,
    private val verbose: Boolean = false,
) {
    /** 토큰 문자열 → ID 매핑. 학습/복원 시 채워짐. */
    private val tokenToId = mutableMapOf<String, Int>()

    /** 병합 규칙 (순서 중요 — 적용 순서대로). */
    private val merges = mutableListOf<TokenPair>()

    fun train(text: String) {
        log("BPE 학습 시작 (목표 어휘 크기: $maxVocabSize, 텍스트 길이: ${text.length})")
        val startTime = System.currentTimeMillis()

        // 1) 정규화 + 특수 토큰 주변 공백 추가
        var processedText = if (lowercase) text.lowercase() else text
        specialTokens.forEachIndexed { index, token ->
            tokenToId[token] = index
            processedText = processedText.replace(token, " $token ")
        }
        log("예약된 특수 토큰: $specialTokens")

        // 2) 모든 고유 문자를 어휘에 추가
        val uniqueChars = processedText.toSet().sorted()
        for (char in uniqueChars) {
            val s = char.toString()
            if (s !in tokenToId) tokenToId[s] = tokenToId.size
        }
        log("총 고유 문자 수: ${uniqueChars.size}")

        // 3) 단어별로 토큰화 (preTokenize off면 전체가 단어 하나)
        val words: MutableList<MutableList<String>> = splitToWords(processedText).toMutableList()
        log("초기 단어 수: ${words.size}, 전체 토큰 수: ${words.sumOf { it.size }}")

        // 4) 병합 루프
        var iteration = 0
        while (tokenToId.size < maxVocabSize) {
            val pairs = countPairs(words)
            if (pairs.isEmpty()) {
                log("더 이상 병합할 쌍이 없습니다. (반복: $iteration, 어휘 크기: ${tokenToId.size})")
                break
            }
            val bestPair = selectBestPair(pairs)
            val mergedToken = bestPair.toMergedToken()

            tokenToId[mergedToken] = tokenToId.size
            merges += bestPair
            applyMerge(words, bestPair)

            if (verbose && iteration++ % 100 == 0) {
                logIntermediate(startTime, iteration, pairs, words)
            }
        }

        logFinal(startTime, processedText, words)
    }

    /** 텍스트를 토큰 ID 리스트로 인코딩. 학습과 완전히 동일한 규칙 적용. */
    fun encode(text: String): List<Int> {
        if (text.isEmpty()) return emptyList()

        var processedText = if (lowercase) text.lowercase() else text
        specialTokens.forEach { token ->
            processedText = processedText.replace(token, " $token ")
        }

        // 학습과 같은 방식으로 단어 분할 + 초기 토큰화
        val words = splitToWords(processedText).toMutableList()

        // 학습된 병합 규칙을 순서대로 적용 (단어 경계는 유지됨)
        for ((index, mergeRule) in merges.withIndex()) {
            applyMerge(words, mergeRule)
            if (verbose && index % 200 == 0) {
                log("  병합 $index/${merges.size} — 현재 토큰 수 ${words.sumOf { it.size }}")
            }
        }

        // ID 변환 (unknown은 UNK ID로 폴백)
        val unknownId = tokenToId[UNKNOWN_TOKEN] ?: 1
        val flatTokens = words.flatten()
        val result = ArrayList<Int>(flatTokens.size)
        var unknownCount = 0
        for (token in flatTokens) {
            val id = tokenToId[token]
            if (id != null) {
                result += id
            } else {
                result += unknownId
                unknownCount++
            }
        }
        if (verbose && unknownCount > 0) {
            log("Unknown 토큰 수: $unknownCount / ${flatTokens.size}")
        }
        return result
    }

    fun getVocabSize(): Int = tokenToId.size

    fun getStoi(): Map<String, Int> = tokenToId.toMap()

    fun getItos(): Map<Int, String> =
        tokenToId.entries.associate { (token, id) -> id to token }

    /** 학습된 병합 규칙을 (first, second) pair 순서대로 반환 — 직렬화용. */
    fun getMerges(): List<Pair<String, String>> =
        merges.map { it.first to it.second }

    /** 외부에서 학습된 상태를 주입해 현재 인스턴스를 복원. Sampler가 학습 시와 같은 토큰화를 재생할 때 사용. */
    fun restore(stoi: Map<String, Int>, merges: List<Pair<String, String>>) {
        tokenToId.clear()
        tokenToId.putAll(stoi)
        this.merges.clear()
        this.merges.addAll(merges.map { TokenPair(it.first, it.second) })
    }

    // =========================================================================
    // 내부 유틸
    // =========================================================================

    /**
     * 전처리된 텍스트를 "단어"들로 쪼개고, 각 단어를 다시 초기 토큰(대부분 단일 문자)으로 분해한다.
     *
     * - [useWordPreTokenize]가 true이면 공백 기반 regex로 `\s*\S+` 형태의 청크로 먼저 나눈다 (GPT-2 스타일).
     *   공백 접두어 단어 " the", " and" 등이 자연스럽게 보존된다.
     * - false면 텍스트 전체를 **한 개의 단어**로 본다 (기존 구현과 호환).
     *
     * 특수 토큰은 `tokenize` 단에서 longest-match로 잡아 온전한 단일 토큰으로 추가한다.
     */
    private fun splitToWords(text: String): List<MutableList<String>> {
        val chunks: List<String> = if (useWordPreTokenize) {
            // \s*\S+ 는 "선택적 선행 공백 + 연속된 비공백" 즉 단어 덩어리 하나
            Regex("\\s*\\S+").findAll(text).map { it.value }.toList()
        } else {
            listOf(text)
        }
        return chunks.map { tokenize(it) }
    }

    /** 한 "단어" 청크를 초기 토큰(특수 토큰 longest-match + 나머지는 char 단위)으로 분해. */
    private fun tokenize(chunk: String): MutableList<String> {
        val tokens = mutableListOf<String>()
        val sortedSpecialTokens = specialTokens.sortedByDescending { it.length }
        var i = 0
        while (i < chunk.length) {
            var matched = false
            for (special in sortedSpecialTokens) {
                if (i + special.length <= chunk.length && chunk.startsWith(special, i)) {
                    tokens += special
                    i += special.length
                    matched = true
                    break
                }
            }
            if (!matched) {
                tokens += chunk[i].toString()
                i++
            }
        }
        return tokens
    }

    /** 단어별 바이그램 빈도를 센다. 특수 토큰은 병합 대상에서 제외. */
    private fun countPairs(words: List<List<String>>): Map<TokenPair, Int> {
        val pairs = HashMap<TokenPair, Int>(1024)
        val specialSet = specialTokens.toHashSet()
        for (word in words) {
            for (i in 0 until word.size - 1) {
                val first = word[i]
                val second = word[i + 1]
                if (first in specialSet || second in specialSet) continue
                pairs.merge(TokenPair(first, second), 1, Int::plus)
            }
        }
        return pairs
    }

    /** 각 단어 안에서 주어진 bigram 쌍을 병합. specialToken은 건너뜀. */
    private fun applyMerge(words: MutableList<MutableList<String>>, pair: TokenPair) {
        val specialSet = specialTokens.toHashSet()
        for (wi in words.indices) {
            val word = words[wi]
            val merged = ArrayList<String>(word.size)
            var i = 0
            while (i < word.size) {
                if (i < word.size - 1 &&
                    word[i] == pair.first && word[i + 1] == pair.second &&
                    word[i] !in specialSet && word[i + 1] !in specialSet
                ) {
                    merged += pair.toMergedToken()
                    i += 2
                } else {
                    merged += word[i]
                    i++
                }
            }
            words[wi] = merged.toMutableList()
        }
    }

    /** 병합할 쌍 선택. [standardBpeScoring]이 true이면 순수 빈도 최대, false면 레거시 휴리스틱. */
    private fun selectBestPair(pairs: Map<TokenPair, Int>): TokenPair {
        if (standardBpeScoring) {
            return pairs.maxByOrNull { it.value }!!.key
        }
        // 레거시: 빈도 × (길이 보너스) × (공백 페널티)
        return pairs.maxByOrNull { (pair, frequency) ->
            var score = frequency.toDouble()
            val totalLength = pair.first.length + pair.second.length
            if (totalLength <= 6) score *= (1.0 + totalLength * 0.1)
            val spaceCount = pair.first.count { it == ' ' } + pair.second.count { it == ' ' }
            if (spaceCount > 0) score *= (1.0 - spaceCount * 0.2)
            score
        }!!.key
    }

    // =========================================================================
    // 로깅
    // =========================================================================

    private fun log(message: String) {
        if (verbose) println(message)
    }

    private fun logIntermediate(
        startTime: Long,
        iteration: Int,
        pairs: Map<TokenPair, Int>,
        words: List<List<String>>,
    ) {
        val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
        val progress = (tokenToId.size.toDouble() / maxVocabSize * 100).toInt()
        println("\n=== 진행 상황 리포트 ===")
        println("$iteration 번 병합 완료 (어휘 ${tokenToId.size}/$maxVocabSize = $progress%)")
        println("총 bigram 종류: ${pairs.size}, 전체 토큰 수: ${words.sumOf { it.size }}")
        pairs.toList()
            .sortedByDescending { it.second }
            .take(5)
            .forEach { (pair, count) ->
                println("  (${pair.first} + ${pair.second}) × $count")
            }
        println("소요 시간: ${elapsed}s, 속도: ${String.format("%.2f", iteration / elapsed)} 병합/초")
    }

    private fun logFinal(
        startTime: Long,
        processedText: String,
        words: List<List<String>>,
    ) {
        val totalTime = (System.currentTimeMillis() - startTime) / 1000.0
        val tokenCount = words.sumOf { it.size }
        val compressionPercent = "%.2f".format(tokenCount.toDouble() / processedText.length * 100)
        println("\n=== BPE 학습 완료 ===")
        println("최종 어휘 크기: ${tokenToId.size}/$maxVocabSize")
        println("병합 횟수: ${merges.size}")
        println("최종 토큰 수: $tokenCount (원본 문자 수 대비 $compressionPercent%)")
        println("총 소요 시간: ${totalTime}s")
        val longTokens = tokenToId.keys.filter { it !in specialTokens }
            .sortedByDescending { it.length }.take(10)
        println("가장 긴 토큰 상위 10개: ${longTokens.joinToString { "'$it'" }}")
    }

    /** 두 토큰의 쌍. */
    data class TokenPair(val first: String, val second: String) {
        fun toMergedToken(): String = "$first$second"
    }

    companion object {
        /** 어휘에 없는 토큰에 할당되는 특수 토큰. id 1로 고정되는 것은 기본 specialTokens 순서에 의존. */
        const val UNKNOWN_TOKEN = "<|unk|>"

        /** 문서/샘플 시작을 표시하는 특수 토큰. 기본 배치에서 id 2. */
        const val BOS_TOKEN = "<|bos|>"

        /** 문서/샘플 끝을 표시하는 특수 토큰. 기본 배치에서 id 0 — Sampler의 stop 조건으로 사용. */
        const val EOS_TOKEN = "<|eos|>"
    }
}
