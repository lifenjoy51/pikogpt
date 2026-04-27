package vec

import data.MetaInfo
import data.SimpleBPE
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import sample.SampleConfig
import vec.layer.VecPikoGPT
import java.io.File
import java.nio.ByteBuffer
import kotlin.math.exp
import kotlin.math.ln
import kotlin.random.Random

/**
 * 벡터 백엔드 텍스트 생성기.
 *
 * 스칼라 `sample.ScalarSampler`와 유사하지만 모델이 `vec.layer.VecPikoGPT`이고 forward가
 * 그래프를 만들지 않으므로 no-grad 컨텍스트 같은 장치가 필요 없다.
 *
 * 체크포인트 레이아웃 (VecTrainer가 만드는 것 그대로):
 *   - `checkpoint.json`   VecCheckpoint
 *   - `model_weights.bin` float32 big-endian 연속
 *   - `meta.json`         vocab + BPE merges
 */
class VecSampler(private val samplingConfig: SampleConfig) {

    private val model: VecPikoGPT
    private val encode: (String) -> List<Int>
    private val decode: (List<Int>) -> String
    private val vocabSize: Int
    private val blockSize: Int

    /** 시드된 RNG (재현성을 위해 Random.Default와 분리). */
    private val rng: Random = Random(samplingConfig.randomSeed)

    init {

        // 1) 체크포인트 메타 로드 (모델 아키텍처 복원용)
        val checkpointFile = File("${samplingConfig.modelDirectoryPath}/checkpoint.json")
        require(checkpointFile.exists()) { "체크포인트 없음: ${checkpointFile.absolutePath}" }
        val parser = Json { ignoreUnknownKeys = true }
        val meta = parser.decodeFromString<VecCheckpoint>(checkpointFile.readText())

        // 2) 모델 구성 + 가중치 로드
        model = VecPikoGPT(meta.modelArgs)
        // 샘플링은 항상 추론 모드 — dropout 비활성.
        model.setTraining(false)
        loadWeights(File("${samplingConfig.modelDirectoryPath}/model_weights.bin"))
        vocabSize = meta.modelArgs.vocabularySize
        blockSize = meta.modelArgs.maxSequenceLength
        println("# 모델 로드 완료 (iter=${meta.iterationNumber}, val loss=${meta.bestValidationLoss})")

        // 3) 토크나이저 복원 (meta.json → SimpleBPE.restore 경로)
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

    /**
     * 누적된 prompt token id list에서 새 토큰만 생성·디코드해 반환.
     * Chat REPL에서 이전 turn 누적 context를 그대로 받아 다음 turn만 생성하기 위한 용도.
     *
     * 반환값 = (새로 생성된 token id 리스트, 디코드된 문자열). stop 토큰은 포함 안 됨.
     */
    fun continueOne(promptIds: IntArray): Pair<List<Int>, String> {
        val stopSet = samplingConfig.stopTokenIds.toHashSet()
        val full = generateTokenSequence(promptIds.toList())
        val newIds = full.drop(promptIds.size).takeWhile { it !in stopSet }
        return newIds to decode(newIds)
    }

    /** 외부에서 텍스트 → 토큰 id 인코딩이 필요할 때 (Chat 등). */
    fun encodeText(text: String): List<Int> = encode(text)

    /** Chat용 max context 길이 — model의 blockSize와 동일. */
    val maxContextLength: Int get() = blockSize

    // =========================================================================
    // 내부
    // =========================================================================

    private fun generateTokenSequence(contextIds: List<Int>): List<Int> {
        val seq = contextIds.toMutableList()
        var ctx = contextIds.toIntArray()

        repeat(samplingConfig.maximumNewTokens) {
            if (ctx.size > blockSize) ctx = ctx.takeLast(blockSize).toIntArray()

            val logits = model.forward(ctx)
            val t = logits.rows
            val v = logits.cols
            // 마지막 위치의 logit row
            val lastLogits = FloatArray(v)
            for (j in 0 until v) lastLogits[j] = logits.data[(t - 1) * v + j]

            // 1) Repetition penalty — 직전 window 내 등장 토큰 logit 약화. temperature 전 단계.
            //   표준 공식: logit > 0 → /penalty, logit < 0 → *penalty.
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

            // 2) Temperature scaling
            val temp = samplingConfig.samplingTemperature
            if (temp != 1.0f && temp > 0.0f) {
                for (j in 0 until v) lastLogits[j] /= temp
            }

            // 3) Top-k 마스킹
            val topK = samplingConfig.topKFilteringSize
            if (topK in 1 until v) {
                val kth = lastLogits.toList()
                    .sortedByDescending { it }[topK - 1]
                for (j in 0 until v) if (lastLogits[j] < kth) lastLogits[j] = Float.NEGATIVE_INFINITY
            }

            // 4) Top-p (nucleus) 마스킹 — 정렬 후 누적 확률이 p 넘는 시점부터 cutoff.
            //   top-k와 병행 시 둘 다 적용된 후보 안에서 다시 nucleus 적용.
            val topP = samplingConfig.topProbabilityThreshold
            if (topP > 0f && topP < 1.0f) {
                // -inf 제외하고 (idx, value) 정렬 → softmax → 누적 → cutoff
                val sortedIdx = (0 until v).sortedByDescending { lastLogits[it] }
                // softmax temporarily on lastLogits (clone) — for cumulative thresholding
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

            // softmax + 샘플링
            val probs = softmaxInPlace(lastLogits)
            val chosen = sampleFromDistribution(probs)
            seq += chosen
            ctx = ctx + chosen
        }
        return seq
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
            val bpe = SimpleBPE(
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
            // 구버전 meta에는 merges가 없으니 greedy 폴백
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
                tokens += 1  // UNK
                i++
            }
        }
        return tokens
    }
}
