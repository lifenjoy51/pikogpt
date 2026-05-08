package turbo

import data.MetaInfo
import data.CharBPE
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import sample.SampleConfig
import turbo.layer.TurboPikoGPT
import java.io.File
import java.nio.ByteBuffer
import kotlin.math.exp
import kotlin.random.Random

/**
 * turbo 백엔드 텍스트 생성기.
 *   - Phase 0: vec.VecSampler와 동등 (naive forward).
 *   - Phase 3.0: useKvCache=true(default)이고 모델이 단순 모드일 때 KV cache로 토큰당 비용 가속.
 *
 * 단순 모드 = numKvHeads=numHeads, !useFusedQkv, !useQkNorm. 옵션 결합 모델은 자동 fallback to naive.
 * 또 prompt + maxNewTokens > blockSize 시에도 fallback (KV cache 슬라이딩 미지원).
 */
class TurboSampler(
    private val samplingConfig: SampleConfig,
    private val useKvCache: Boolean = true,
) {

    private val model: TurboPikoGPT
    private val encode: (String) -> List<Int>
    private val decode: (List<Int>) -> String
    private val vocabSize: Int
    private val blockSize: Int

    private val rng: Random = Random(samplingConfig.randomSeed)

    init {
        val checkpointFile = File("${samplingConfig.modelDirectoryPath}/checkpoint.json")
        require(checkpointFile.exists()) { "체크포인트 없음: ${checkpointFile.absolutePath}" }
        val parser = Json { ignoreUnknownKeys = true }
        val meta = parser.decodeFromString<TurboCheckpoint>(checkpointFile.readText())

        model = TurboPikoGPT(meta.modelArgs)
        model.setTraining(false)
        loadWeights(File("${samplingConfig.modelDirectoryPath}/model_weights.bin"))
        vocabSize = meta.modelArgs.gpt.vocabularySize
        blockSize = meta.modelArgs.gpt.maxSequenceLength
        println("# 모델 로드 완료 (iter=${meta.iterationNumber}, val loss=${meta.bestValidationLoss})")

        val metaInfo = parser.decodeFromString<MetaInfo>(
            File("${samplingConfig.modelDirectoryPath}/meta.json").readText()
        )
        val (enc, dec) = buildEncoderDecoder(metaInfo)
        encode = enc
        decode = dec
    }

    fun generate(prompt: String): List<String> {
        val initialIds = encode(prompt)
        println("# 프롬프트 '${prompt}' → 토큰 ${initialIds.size}개")

        val stopSet = samplingConfig.stopTokenIds.toHashSet()
        val samples = (0 until samplingConfig.numberOfSamples).map { _ ->
            val generated = generateTokenSequence(initialIds).takeWhile { it !in stopSet }
            decode(generated)
        }
        return samples
    }

    fun continueOne(promptIds: IntArray): Pair<List<Int>, String> {
        val stopSet = samplingConfig.stopTokenIds.toHashSet()
        val full = generateTokenSequence(promptIds.toList())
        val newIds = full.drop(promptIds.size).takeWhile { it !in stopSet }
        return newIds to decode(newIds)
    }

    fun encodeText(text: String): List<Int> = encode(text)

    val maxContextLength: Int get() = blockSize

    private fun generateTokenSequence(contextIds: List<Int>): List<Int> {
        val totalNeeded = contextIds.size + samplingConfig.maximumNewTokens
        val canUseKvCache = useKvCache && isSimpleMode() && totalNeeded <= blockSize
        return if (canUseKvCache) generateTokenSequenceWithKvCache(contextIds)
               else generateTokenSequenceNaive(contextIds)
    }

    /** 모델이 KV cache 단순 모드 (Phase 3.0)에 해당하는지 — Phase 3.x에서 옵션 확장 시 완화. */
    private fun isSimpleMode(): Boolean {
        val cfg = model.config
        val mhaOk = cfg.numKvHeads == null || cfg.numKvHeads == cfg.gpt.numberOfAttentionHeads
        return mhaOk && !cfg.useFusedQkv && !cfg.useQkNorm
    }

    private fun generateTokenSequenceNaive(contextIds: List<Int>): List<Int> {
        val seq = contextIds.toMutableList()
        var ctx = contextIds.toIntArray()

        repeat(samplingConfig.maximumNewTokens) {
            if (ctx.size > blockSize) ctx = ctx.takeLast(blockSize).toIntArray()

            val logits = model.forward(ctx)
            val t = logits.rows
            val v = logits.cols
            val lastLogits = FloatArray(v)
            for (j in 0 until v) lastLogits[j] = logits.data[(t - 1) * v + j]

            val chosen = sampleNextTokenFromLogits(lastLogits, seq, v)
            seq += chosen
            ctx = ctx + chosen
        }
        return seq
    }

    private fun generateTokenSequenceWithKvCache(contextIds: List<Int>): List<Int> {
        val seq = contextIds.toMutableList()
        val gpt = model.config.gpt
        val headDim = gpt.embeddingDimension / gpt.numberOfAttentionHeads
        val kvDim = model.config.effectiveKvHeads * headDim
        val cache = TurboKVCache(
            maxSeqLen = blockSize,
            numLayers = gpt.numberOfLayers,
            kvDim = kvDim,
        )

        // Prompt 단계: 모든 prompt 토큰을 incremental forward (마지막 logits만 사용)
        var lastLogits: TurboTensor? = null
        for (token in contextIds) {
            lastLogits = model.forwardIncremental(token, cache)
        }
        val v = lastLogits?.cols ?: vocabSize

        repeat(samplingConfig.maximumNewTokens) {
            val current = lastLogits ?: error("KV cache 모드: empty prompt 미지원")
            val logitsRow = FloatArray(v)
            for (j in 0 until v) logitsRow[j] = current.data[j]

            val chosen = sampleNextTokenFromLogits(logitsRow, seq, v)
            seq += chosen

            if (cache.length >= blockSize) {
                // cache 가득 — 더 이상 incremental 불가. KV cache 모드 종료.
                return seq
            }
            lastLogits = model.forwardIncremental(chosen, cache)
        }
        return seq
    }

    /**
     * 단일 logits row에 repetition penalty / temperature / top-k / top-p / softmax / 샘플링을
     * 적용해 다음 토큰 id를 반환. naive와 KV cache 두 경로에서 공유.
     */
    private fun sampleNextTokenFromLogits(lastLogits: FloatArray, seq: List<Int>, v: Int): Int {
        val repPen = samplingConfig.repetitionPenalty
        if (repPen != 1.0f && repPen > 0.0f) {
            val window = samplingConfig.repetitionWindow.coerceAtLeast(1)
            val recent = seq.takeLast(window).toHashSet()
            for (id in recent) {
                if (id in 0 until v) {
                    val l = lastLogits[id]
                    lastLogits[id] = if (l > 0f) l / repPen else l * repPen
                }
            }
        }

        val temp = samplingConfig.samplingTemperature
        if (temp != 1.0f && temp > 0.0f) {
            for (j in 0 until v) lastLogits[j] /= temp
        }

        val topK = samplingConfig.topKFilteringSize
        if (topK in 1 until v) {
            val kth = lastLogits.toList().sortedByDescending { it }[topK - 1]
            for (j in 0 until v) if (lastLogits[j] < kth) lastLogits[j] = Float.NEGATIVE_INFINITY
        }

        val topP = samplingConfig.topProbabilityThreshold
        if (topP > 0f && topP < 1.0f) {
            val sortedIdx = (0 until v).sortedByDescending { lastLogits[it] }
            val maxVal = lastLogits.max()
            if (maxVal != Float.NEGATIVE_INFINITY) {
                var sumExp = 0.0
                val exps = DoubleArray(v)
                for (j in 0 until v) {
                    if (lastLogits[j] == Float.NEGATIVE_INFINITY) {
                        exps[j] = 0.0
                    } else {
                        val e = kotlin.math.exp((lastLogits[j] - maxVal).toDouble())
                        exps[j] = e
                        sumExp += e
                    }
                }
                if (sumExp > 0.0) {
                    var cum = 0.0
                    val keep = BooleanArray(v)
                    for (idx in sortedIdx) {
                        val p = exps[idx] / sumExp
                        cum += p
                        keep[idx] = true
                        if (cum >= topP) break
                    }
                    for (j in 0 until v) if (!keep[j]) lastLogits[j] = Float.NEGATIVE_INFINITY
                }
            }
        }

        val probs = softmaxInPlace(lastLogits)
        return sampleFromDistribution(probs)
    }

    private fun softmaxInPlace(logits: FloatArray): FloatArray {
        var maxVal = Float.NEGATIVE_INFINITY
        for (v in logits) if (v > maxVal) maxVal = v
        var sum = 0.0f
        for (i in logits.indices) {
            val e = exp((logits[i] - maxVal).toDouble()).toFloat()
            logits[i] = e
            sum += e
        }
        val inv = 1.0f / sum
        for (i in logits.indices) logits[i] *= inv
        return logits
    }

    private fun sampleFromDistribution(probs: FloatArray): Int {
        val r = rng.nextDouble().toFloat()
        var cumulative = 0.0f
        for (i in probs.indices) {
            cumulative += probs[i]
            if (r <= cumulative) return i
        }
        return probs.size - 1
    }

    private fun loadWeights(file: File) {
        require(file.exists()) { "가중치 파일 없음: ${file.absolutePath}" }
        file.inputStream().use { input ->
            val buf = ByteArray(4)
            for (p in model.parameters()) {
                for (i in 0 until p.numel) {
                    require(input.read(buf) == 4) { "가중치 파일 EOF 조기 도달" }
                    p.data[i] = ByteBuffer.wrap(buf).float
                }
            }
        }
    }

    private fun buildEncoderDecoder(metaInfo: MetaInfo): Pair<(String) -> List<Int>, (List<Int>) -> String> {
        val encoder: (String) -> List<Int> = if (metaInfo.merges.isNotEmpty()) {
            val bpe = CharBPE(
                maxVocabSize = metaInfo.vocabularySize,
                specialTokens = metaInfo.specialTokens,
                lowercase = metaInfo.lowercase,
                useWordPreTokenize = metaInfo.useWordPreTokenize,
                standardBpeScoring = true,
                verbose = false,
            )
            val mergePairs = metaInfo.merges.map { it[0] to it[1] }
            bpe.restore(metaInfo.stringToIndex, mergePairs)
            ({ inputText -> bpe.encode(inputText) })
        } else {
            ({ inputText -> greedyTokenize(inputText, metaInfo.stringToIndex) })
        }

        val specialSet = metaInfo.specialTokens.toHashSet()
        val decoder: (List<Int>) -> String = { ids ->
            ids.mapNotNull { id -> metaInfo.indexToString[id]?.takeUnless { it in specialSet } }
                .joinToString("")
        }

        return encoder to decoder
    }

    private fun greedyTokenize(text: String, stoi: Map<String, Int>): List<Int> {
        val tokens = mutableListOf<Int>()
        val longest = stoi.keys.maxOf { it.length }
        var i = 0
        while (i < text.length) {
            var matched = false
            for (length in minOf(text.length - i, longest) downTo 1) {
                val candidate = text.substring(i, i + length)
                val id = stoi[candidate]
                if (id != null) {
                    tokens += id
                    i += length
                    matched = true
                    break
                }
            }
            if (!matched) {
                tokens += 1
                i++
            }
        }
        return tokens
    }
}
