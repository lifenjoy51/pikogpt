package data

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.runBlocking

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
class CharBPE(
    private val maxVocabSize: Int,
    private val specialTokens: List<String> = listOf(EOS_TOKEN, UNKNOWN_TOKEN, BOS_TOKEN, TURN_TOKEN, SEP_TOKEN),
    private val lowercase: Boolean = false,
    private val useWordPreTokenize: Boolean = false,
    private val standardBpeScoring: Boolean = true,
    private val verbose: Boolean = false,
    /**
     * 공백을 BPE merge 대상에서 제외하고 단일 ' ' 토큰으로 고정한다.
     * true면 splitToWords가 공백 chunk와 비공백 chunk를 분리하고, 공백 chunk는 char 분해 없이
     * 단일 토큰으로 처리됨 → BPE는 알파벳/숫자만 merge.
     * useWordPreTokenize와 동시에 사용 시 의도가 모호하므로 둘 중 하나만 권장.
     */
    private val splitSpaceAsToken: Boolean = false,
    /**
     * BPE merge 결과 토큰의 최대 char 길이. null이면 무제한(표준 BPE).
     * 예: 2로 설정하면 bigram(`ar`, `er` 등)까지만 merge하고 그 이상은 skip.
     * 작은 값 + 작은 vocab 조합으로 "char + 자주 나오는 bigram만" 학습하는 단순 토크나이저 구성.
     */
    private val maxTokenLength: Int? = null,
) {
    /** 토큰 문자열 → ID 매핑. 학습/복원 시 채워짐. */
    private val tokenToId = mutableMapOf<String, Int>()

    /** 병합 규칙 (순서 중요 — 적용 순서대로). */
    private val merges = mutableListOf<TokenPair>()

    /**
     * Char → String 캐시. tokenize에서 `chunk[i].toString()`을 매 char마다 새 객체로 만들면
     * 큰 코퍼스(수억 char)에서 OOM 위험. 고유 char은 보통 100개 미만이라 캐시 효과 큼.
     */
    private val charStringCache = HashMap<Char, String>()
    private fun charToString(c: Char): String = charStringCache.getOrPut(c) { c.toString() }

    fun train(text: String) {
        log("BPE 학습 시작 (목표 어휘 크기: $maxVocabSize, 텍스트 길이: ${text.length})")
        val startTime = System.currentTimeMillis()

        // 1) 정규화 + 특수 토큰 vocab 등록 (id 0~). tokenize()가 longest-match로 단위 인식하므로
        //    텍스트에 공백을 추가하지 않는다 — 공백 padding은 ` b`, `'  '` 같은 artifact를 만든다.
        val processedText = if (lowercase) text.lowercase() else text
        specialTokens.forEachIndexed { index, token ->
            tokenToId[token] = index
        }
        log("예약된 특수 토큰: $specialTokens")

        // 2) 모든 고유 문자를 어휘에 추가
        val uniqueChars = processedText.toSet().sorted()
        for (char in uniqueChars) {
            val s = char.toString()
            if (s !in tokenToId) tokenToId[s] = tokenToId.size
        }
        log("총 고유 문자 수: ${uniqueChars.size}")

        // 3) 단어별로 토큰화 후 unique word 빈도 압축
        //
        // Zipf 분포의 자연어에서는 같은 word가 매우 자주 반복된다 ("the" 수십만 번 등).
        // 이 word들을 매 merge round마다 별도로 순회하면 같은 작업이 수십만 번 중복된다.
        // (word, count) 형태로 묶어 unique key만 처리하면 결과는 같고 속도만 30× 이상 빨라진다.
        // 핵심: bigram 빈도는 word_count의 **합**으로 계산되고, merge 결과 word는 동일 키로
        // 다시 합쳐지므로 raw 순회와 수학적으로 동등하다.
        val rawWords = splitToWords(processedText)
        val rawWordCount = rawWords.size
        val rawTokenCount = rawWords.sumOf { it.size }
        var wordCounts: HashMap<List<String>, Long> = HashMap(rawWords.size.coerceAtLeast(1024))
        var skippedWhitespace = 0L
        for (word in rawWords) {
            // splitSpaceAsToken 모드: 공백-only word는 BPE merge 대상에서 제외 → '  ',
            // ' '+letter 같은 멀티공백/혼합 토큰 학습 방지. 기본 char ID로만 인코딩됨.
            if (splitSpaceAsToken && word.all { it.isNotEmpty() && it[0].isWhitespace() }) {
                skippedWhitespace++
                continue
            }
            // MutableList → 불변 List로 변환해 안전한 Map key로 사용
            wordCounts.merge(word.toList(), 1L, Long::plus)
        }
        log("초기 단어 수: $rawWordCount (unique: ${wordCounts.size}, 공백 skip: $skippedWhitespace), 전체 토큰 수: $rawTokenCount")

        // 4) 병합 루프 (unique word 단위)
        var iteration = 0
        while (tokenToId.size < maxVocabSize) {
            val pairs = countPairsCompressed(wordCounts)
            if (pairs.isEmpty()) {
                log("더 이상 병합할 쌍이 없습니다. (반복: $iteration, 어휘 크기: ${tokenToId.size})")
                break
            }
            // maxTokenLength 제한이 있으면 너무 긴 merge 후보 제외.
            val candidatePairs = if (maxTokenLength != null) {
                pairs.filterKeys { it.toMergedToken().length <= maxTokenLength }
            } else pairs
            if (candidatePairs.isEmpty()) {
                log("더 이상 maxTokenLength=$maxTokenLength 이하 merge 후보 없음. (어휘 크기: ${tokenToId.size})")
                break
            }
            val bestPair = selectBestPair(candidatePairs)
            val mergedToken = bestPair.toMergedToken()

            tokenToId[mergedToken] = tokenToId.size
            merges += bestPair
            wordCounts = applyMergeCompressed(wordCounts, bestPair)

            if (verbose && iteration++ % 100 == 0) {
                logIntermediateCompressed(startTime, iteration, pairs, wordCounts)
            }
        }

        logFinalCompressed(startTime, processedText, wordCounts)
    }

    /** 텍스트를 토큰 ID 리스트로 인코딩. 학습과 완전히 동일한 규칙 적용.
     *
     * **Word-level 캐싱 최적화**: 같은 word는 한 번만 merge 처리하고 결과(token ID 리스트)를
     * `HashMap<List<String>, IntArray>`에 캐시. Zipf 분포 자연어에서 unique word 수가 raw word 수의
     * ~1/180 수준이라 cache hit ratio 매우 높아 ~수백× 가속. 결과 token 시퀀스는 동일.
     *
     * 구버전(merge 1996회 전체 word 재처리)은 큰 코퍼스에서 hours 단위로 길어졌음.
     */
    fun encode(text: String): List<Int> {
        if (text.isEmpty()) return emptyList()

        // train()과 동일하게 공백 padding 없이 처리. tokenize()가 special token을 longest-match로 인식.
        val processedText = if (lowercase) text.lowercase() else text

        // 학습과 같은 방식으로 단어 분할 + 초기 토큰화
        val rawWords = splitToWords(processedText)

        val unknownId = tokenToId[UNKNOWN_TOKEN] ?: 1
        val specialSet = specialTokens.toHashSet()
        val cache = HashMap<List<String>, IntArray>()
        var totalTokens = 0
        var unknownCount = 0

        // 1차 패스: unique word만 cache에 채움 (merge 시퀀스 1번씩만 적용)
        // 결과 ID 리스트를 IntArray에 저장.
        for (word in rawWords) {
            val key = word.toList()
            if (key in cache) continue
            val tokens = applyAllMergesToWord(word, specialSet)
            val ids = IntArray(tokens.size) { i ->
                val id = tokenToId[tokens[i]]
                if (id != null) id else {
                    unknownCount++
                    unknownId
                }
            }
            cache[key] = ids
        }

        // 2차 패스: word 순서대로 캐시된 IDs를 result에 누적.
        // 정확한 길이 계산을 위해 한 번 더 순회.
        var totalLen = 0
        for (word in rawWords) totalLen += cache[word.toList()]!!.size
        val result = ArrayList<Int>(totalLen)
        for (word in rawWords) {
            for (id in cache[word.toList()]!!) result += id
        }
        totalTokens = result.size

        if (verbose) {
            log("encode 완료: rawWords=${rawWords.size}, uniqueWords=${cache.size}, tokens=$totalTokens, unknown=$unknownCount")
        }
        return result
    }

    /**
     * 단일 word에 학습된 merge 규칙을 순서대로 적용해 최종 토큰 시퀀스 반환.
     * encode()의 word-cache용 helper. apply 결과는 deterministic.
     */
    private fun applyAllMergesToWord(
        word: List<String>,
        specialSet: Set<String>,
    ): List<String> {
        var current: MutableList<String> = ArrayList(word)
        for (mergeRule in merges) {
            if (current.size < 2) break
            // pair가 word에 존재하는지 빠른 확인
            var hasPair = false
            for (i in 0 until current.size - 1) {
                if (current[i] == mergeRule.first && current[i + 1] == mergeRule.second &&
                    current[i] !in specialSet && current[i + 1] !in specialSet
                ) { hasPair = true; break }
            }
            if (!hasPair) continue
            val merged = ArrayList<String>(current.size)
            var i = 0
            while (i < current.size) {
                if (i < current.size - 1 &&
                    current[i] == mergeRule.first && current[i + 1] == mergeRule.second &&
                    current[i] !in specialSet && current[i + 1] !in specialSet
                ) {
                    merged += mergeRule.toMergedToken()
                    i += 2
                } else {
                    merged += current[i]
                    i++
                }
            }
            current = merged
        }
        return current
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
        val chunks: List<String> = when {
            splitSpaceAsToken -> {
                // \s+ 또는 \S+ 매치 → 공백 chunk와 비공백 chunk 분리.
                // 공백 chunk는 char 분해 없이 통째로 1 토큰으로 보존.
                Regex("\\s+|\\S+").findAll(text).map { it.value }.toList()
            }
            useWordPreTokenize -> {
                // \s*\S+ 는 "선택적 선행 공백 + 연속된 비공백" 즉 단어 덩어리 하나
                Regex("\\s*\\S+").findAll(text).map { it.value }.toList()
            }
            else -> listOf(text)
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
                tokens += charToString(chunk[i])
                i++
            }
        }
        return tokens
    }

    /**
     * Encode 시 단어 순서 보존을 위한 raw 버전 — `MutableList<MutableList<String>>`에 in-place 변환.
     * 학습에는 압축 버전(`applyMergeCompressed`)을 쓰지만, encode()는 결과 토큰 순서가 입력 순서와
     * 같아야 하므로 raw 자료구조를 그대로 사용.
     */
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

    /**
     * 병렬 worker 수. 환경변수 `BPE_MAX_WORKERS`로 override 가능.
     * 기본은 가용 CPU 수(상한 8). countPairs / applyMerge 모두 사용.
     */
    private val parallelism: Int by lazy {
        val cpu = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)
        val envCap = System.getenv("BPE_MAX_WORKERS")?.toIntOrNull()?.coerceAtLeast(1)
        val cap = envCap ?: cpu.coerceAtMost(8)
        cap
    }

    /**
     * Unique word 묶음에서 bigram 빈도를 센다. 특수 토큰은 병합 대상 제외.
     *
     * 같은 word가 N번 등장하면 그 안의 bigram들도 각각 N번 카운트되어야 정확하므로
     * `freq`(해당 word의 raw 등장 수)를 그대로 더한다. raw 단어 리스트를 순회하던
     * 이전 구현과 수학적으로 동등.
     *
     * **병렬화**: word entries를 N개 chunk로 분할 → 각 worker가 local HashMap에 누적
     * → 마지막에 partial map들 합산. word 처리는 독립적이라 race 없음.
     */
    private fun countPairsCompressed(counts: Map<List<String>, Long>): Map<TokenPair, Long> {
        val specialSet = specialTokens.toHashSet()
        val entries = counts.entries.toList()
        val chunkSize = ((entries.size + parallelism - 1) / parallelism).coerceAtLeast(1)
        val chunks = entries.chunked(chunkSize)

        val partials = if (chunks.size <= 1) {
            listOf(countPairsLocal(chunks.firstOrNull() ?: emptyList(), specialSet))
        } else {
            runBlocking {
                coroutineScope {
                    chunks.map { chunk ->
                        async(Dispatchers.Default) { countPairsLocal(chunk, specialSet) }
                    }.awaitAll()
                }
            }
        }

        // 빠른 합산: 가장 큰 partial을 reuse해서 나머지 더함
        val combined = partials.maxByOrNull { it.size } ?: HashMap()
        for (p in partials) {
            if (p === combined) continue
            for ((k, v) in p) combined.merge(k, v, Long::plus)
        }
        return combined
    }

    private fun countPairsLocal(
        chunk: List<Map.Entry<List<String>, Long>>,
        specialSet: Set<String>,
    ): HashMap<TokenPair, Long> {
        val local = HashMap<TokenPair, Long>(1024)
        for ((word, freq) in chunk) {
            for (i in 0 until word.size - 1) {
                val first = word[i]
                val second = word[i + 1]
                if (first in specialSet || second in specialSet) continue
                local.merge(TokenPair(first, second), freq, Long::plus)
            }
        }
        return local
    }

    /**
     * 압축된 word 빈도 맵에 merge 규칙을 적용. unique word 한 개당 1번만 변환하고
     * 결과 word가 동일 키로 떨어지면 빈도 합산. specialToken은 건너뜀.
     *
     * 빠른 경로: pair가 word에 등장하지 않으면 word를 그대로 새 맵에 복사.
     *
     * **병렬화**: 각 worker가 local out HashMap 만들고 마지막에 합산.
     */
    private fun applyMergeCompressed(
        counts: Map<List<String>, Long>,
        pair: TokenPair,
    ): HashMap<List<String>, Long> {
        val specialSet = specialTokens.toHashSet()
        val mergedToken = pair.toMergedToken()
        val entries = counts.entries.toList()
        val chunkSize = ((entries.size + parallelism - 1) / parallelism).coerceAtLeast(1)
        val chunks = entries.chunked(chunkSize)

        val partials = if (chunks.size <= 1) {
            listOf(applyMergeLocal(chunks.firstOrNull() ?: emptyList(), pair, mergedToken, specialSet))
        } else {
            runBlocking {
                coroutineScope {
                    chunks.map { chunk ->
                        async(Dispatchers.Default) {
                            applyMergeLocal(chunk, pair, mergedToken, specialSet)
                        }
                    }.awaitAll()
                }
            }
        }

        val combined = partials.maxByOrNull { it.size } ?: HashMap()
        for (p in partials) {
            if (p === combined) continue
            for ((k, v) in p) combined.merge(k, v, Long::plus)
        }
        return combined
    }

    private fun applyMergeLocal(
        chunk: List<Map.Entry<List<String>, Long>>,
        pair: TokenPair,
        mergedToken: String,
        specialSet: Set<String>,
    ): HashMap<List<String>, Long> {
        val out = HashMap<List<String>, Long>(chunk.size)
        for ((word, freq) in chunk) {
            if (!containsPair(word, pair, specialSet)) {
                out.merge(word, freq, Long::plus)
                continue
            }
            val merged = ArrayList<String>(word.size)
            var i = 0
            while (i < word.size) {
                if (i < word.size - 1 &&
                    word[i] == pair.first && word[i + 1] == pair.second &&
                    word[i] !in specialSet && word[i + 1] !in specialSet
                ) {
                    merged += mergedToken
                    i += 2
                } else {
                    merged += word[i]
                    i++
                }
            }
            out.merge(merged, freq, Long::plus)
        }
        return out
    }

    private fun containsPair(word: List<String>, pair: TokenPair, specialSet: Set<String>): Boolean {
        for (i in 0 until word.size - 1) {
            if (word[i] == pair.first && word[i + 1] == pair.second &&
                word[i] !in specialSet && word[i + 1] !in specialSet
            ) return true
        }
        return false
    }

    /** 병합할 쌍 선택. [standardBpeScoring]이 true이면 순수 빈도 최대, false면 레거시 휴리스틱. */
    private fun selectBestPair(pairs: Map<TokenPair, Long>): TokenPair {
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

    private fun logIntermediateCompressed(
        startTime: Long,
        iteration: Int,
        pairs: Map<TokenPair, Long>,
        counts: Map<List<String>, Long>,
    ) {
        val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
        val progress = (tokenToId.size.toDouble() / maxVocabSize * 100).toInt()
        val totalTokens = counts.entries.sumOf { (word, freq) -> word.size.toLong() * freq }
        println("\n=== 진행 상황 리포트 ===")
        println("$iteration 번 병합 완료 (어휘 ${tokenToId.size}/$maxVocabSize = $progress%)")
        println("총 bigram 종류: ${pairs.size}, unique words: ${counts.size}, 전체 토큰 수: $totalTokens")
        pairs.toList()
            .sortedByDescending { it.second }
            .take(5)
            .forEach { (pair, count) ->
                println("  (${pair.first} + ${pair.second}) × $count")
            }
        println("소요 시간: ${elapsed}s, 속도: ${String.format("%.2f", iteration / elapsed)} 병합/초")
    }

    private fun logFinalCompressed(
        startTime: Long,
        processedText: String,
        counts: Map<List<String>, Long>,
    ) {
        val totalTime = (System.currentTimeMillis() - startTime) / 1000.0
        val tokenCount = counts.entries.sumOf { (word, freq) -> word.size.toLong() * freq }
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

        /** 대화 turn 경계를 표시하는 특수 토큰. 기본 배치에서 id 3 — 발화 사이마다 삽입해
         *  모델이 turn-taking 구조를 학습하게 한다. Sampler에서 single-turn 응답을 받으려면
         *  이 토큰에서 stop. */
        const val TURN_TOKEN = "<|turn|>"

        /** 문장/줄 경계를 표시하는 특수 토큰. 기본 배치에서 id 4 — 데이터 prep에서 literal `\n`
         *  같은 sentence delimiter를 단일 토큰으로 박아 BPE merge가 cross-boundary 합성 토큰을
         *  만들지 않게 한다. */
        const val SEP_TOKEN = "<|sep|>"
    }
}
