package vec

import data.MetaInfo
import data.SimpleBPE
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import sample.SampleConfig
import vec.layer.PikoGPT
import java.io.File
import java.nio.ByteBuffer
import kotlin.math.exp
import kotlin.math.ln
import kotlin.random.Random

/**
 * 벡터 백엔드 텍스트 생성기.
 *
 * 스칼라 `sample.Sampler`와 유사하지만 모델이 `vec.layer.PikoGPT`이고 forward가
 * 그래프를 만들지 않으므로 no-grad 컨텍스트 같은 장치가 필요 없다.
 *
 * 체크포인트 레이아웃 (VTrainer가 만드는 것 그대로):
 *   - `checkpoint.json`   VCheckpointMeta
 *   - `model_weights.bin` float32 big-endian 연속
 *   - `meta.json`         vocab + BPE merges
 */
class Sampler(private val samplingConfig: SampleConfig) {

    private val model: PikoGPT
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
        val meta = parser.decodeFromString<VCheckpointMeta>(checkpointFile.readText())

        // 2) 모델 구성 + 가중치 로드
        model = PikoGPT(meta.modelArgs)
        loadWeights(File("${samplingConfig.modelDirectoryPath}/model_weights.bin"))
        vocabSize = meta.modelArgs.vocabSize
        blockSize = meta.modelArgs.blockSize
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

        val samples = (0 until samplingConfig.numSamples).map { _ ->
            val generated = generateTokenSequence(initialIds).takeWhile { it != 0 }  // 0 = EOS
            decode(generated)
        }
        return samples
    }

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

            // temperature scaling
            val temp = samplingConfig.samplingTemperature
            if (temp != 1.0f && temp > 0.0f) {
                for (j in 0 until v) lastLogits[j] /= temp
            }

            // top-k 마스킹
            val topK = samplingConfig.topKFilteringSize
            if (topK in 1 until v) {
                val kth = lastLogits.toList()
                    .sortedByDescending { it }[topK - 1]
                for (j in 0 until v) if (lastLogits[j] < kth) lastLogits[j] = Float.NEGATIVE_INFINITY
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
                maxVocabSize = metaInfo.vocabSize,
                specialTokens = metaInfo.specialTokens,
                lowercase = metaInfo.lowercase,
                useWordPreTokenize = metaInfo.useWordPreTokenize,
                standardBpeScoring = true,
                verbose = false,
            )
            val mergePairs = metaInfo.merges.map { it[0] to it[1] }
            bpe.restore(metaInfo.stoi, mergePairs)
            ({ inputText -> bpe.encode(inputText) })
        } else {
            // 구버전 meta에는 merges가 없으니 greedy 폴백
            ({ inputText -> greedyTokenize(inputText, metaInfo.stoi) })
        }

        val specialSet = metaInfo.specialTokens.toHashSet()
        val decoder: (List<Int>) -> String = { ids ->
            ids.mapNotNull { id -> metaInfo.itos[id]?.takeUnless { it in specialSet } }
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
