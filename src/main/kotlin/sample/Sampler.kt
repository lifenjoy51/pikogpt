package sample

import Value
import data.MetaInfo
import gpt.PikoGPT
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import train.Checkpoint
import java.io.File
import java.nio.ByteBuffer
import kotlin.math.exp
import kotlin.random.Random

// 샘플링 클래스
class Sampler(private val config: SampleConfig) {
    private lateinit var model: PikoGPT
    private lateinit var encode: (String) -> List<Int>
    private lateinit var decode: (List<Int>) -> String
    private var vocabSize: Int = 0

    init {
        Random(config.seed)
    }

    fun sample() {
        println("=== MiniGPT 텍스트 생성 ===")
        println("설정: $config")

        // 모델 로드
        loadModel()

        // 인코더/디코더 설정
        setupEncoding()

        // 시작 텍스트 준비
        val startText = if (config.start.startsWith("FILE:")) {
            File(config.start.substring(5)).readText()
        } else {
            config.start
        }

        println("시작 텍스트: \"$startText\"")
        val startIds = encode(startText)
        println("시작 토큰: $startIds")

        // 샘플 생성
        for (k in 0 until config.numSamples) {
            println("\n--- 샘플 ${k + 1} ---")
            val generated = generate(
                startIds,
                config.maxNewTokens,
                config.temperature,
                config.topK
            )
            println(decode(generated))
        }
    }

    private fun loadModel() {
        when (config.initFrom) {
            "resume" -> {
                println("체크포인트에서 모델 로드 중...")

                val checkpointFile = File("${config.outDir}/checkpoint.json")
                if (!checkpointFile.exists()) {
                    throw IllegalStateException("체크포인트를 찾을 수 없습니다: ${checkpointFile.absolutePath}")
                }

                // 체크포인트 로드
                val json = Json { ignoreUnknownKeys = true }
                val checkpoint = json.decodeFromString<Checkpoint>(checkpointFile.readText())

                // 모델 생성
                val modelConfig = checkpoint.modelArgs
                model = PikoGPT(modelConfig)
                vocabSize = modelConfig.vocabSize

                // 가중치 로드
                loadModelWeights("${config.outDir}/model_weights.bin")

                println("모델 로드 완료 (iteration: ${checkpoint.iterNum}, val loss: ${checkpoint.bestValLoss})")
            }

            "scratch" -> {
                throw IllegalArgumentException("샘플링을 위해서는 학습된 모델이 필요합니다")
            }
        }
    }

    private fun loadModelWeights(filename: String) {
        val file = File(filename)
        if (!file.exists()) {
            println("경고: 가중치 파일을 찾을 수 없습니다. 랜덤 가중치를 사용합니다.")
            return
        }

        file.inputStream().use { stream ->
            val params = model.parameters()
            val buffer = ByteArray(8)

            params.forEach { param ->
                if (stream.read(buffer) == 8) {
                    param.data = ByteBuffer.wrap(buffer).double
                }
            }
        }

        println("모델 가중치 로드 완료")
    }

    private fun setupEncoding() {
        // meta.json 확인
        val metaPath = File("data/${getDataset()}/meta.json")

        if (metaPath.exists()) {
            println("meta.json에서 인코딩 정보 로드 중...")
            val json = Json { ignoreUnknownKeys = true }
            val meta = json.decodeFromString<MetaInfo>(metaPath.readText())

            encode = { s -> s.map { meta.stoi[it.toString()]!! } }
            decode = { l -> l.joinToString("") { meta.itos[it]!! } }
            vocabSize = meta.vocabSize
        } else {
            println("meta.json을 찾을 수 없습니다. 기본 문자 인코딩 사용...")
            // 간단한 문자 수준 인코딩
            val chars = (0..127).map { it.toChar() }
            val stoi = chars.withIndex().associate { it.value.toString() to it.index }
            val itos = chars.withIndex().associate { it.index to it.value.toString() }

            encode = { s -> s.map { stoi[it.toString()] ?: 0 } }
            decode = { l -> l.joinToString("") { itos[it] ?: "?" } }
        }
    }

    private fun getDataset(): String {
        // 실제로는 체크포인트에서 읽어야 함
        return "shakespeare_char"
    }

    private fun generate(
        contextIds: List<Int>,
        maxNewTokens: Int,
        temperature: Double,
        topK: Int
    ): List<Int> {
        val generated = contextIds.toMutableList()
        var context = contextIds.toIntArray()

        repeat(maxNewTokens) {
            // 컨텍스트가 블록 크기를 초과하면 자르기
            if (context.size > model.config.blockSize) {
                context = context.takeLast(model.config.blockSize).toIntArray()
            }

            // 다음 토큰 예측
            val logits = model.forward(context)
            val lastLogits = logits.last() // 마지막 위치의 로짓

            // 온도 적용
            val scaledLogits = lastLogits.map {
                Value(it.data / temperature)
            }.toTypedArray()

            // Top-k 필터링
            val topKLogits = if (topK > 0 && topK < vocabSize) {
                applyTopK(scaledLogits, topK)
            } else {
                scaledLogits
            }

            // Softmax와 샘플링
            val probs = softmax(topKLogits)
            val nextToken = sampleFromDistribution(probs)

            // 생성된 토큰 추가
            generated.add(nextToken)
            context = context + nextToken

            // 생성 중 출력 (선택사항)
            if ((it + 1) % 50 == 0) {
                print(".")
            }
        }
        println() // 줄바꿈

        return generated
    }

    private fun applyTopK(logits: Array<Value>, k: Int): Array<Value> {
        // 로짓을 값에 따라 정렬하여 상위 k개 선택
        val indexed = logits.withIndex().sortedByDescending { it.value.data }
        val topKIndices = indexed.take(k).map { it.index }.toSet()

        // 상위 k개가 아닌 로짓은 -무한대로 설정
        return logits.mapIndexed { i, logit ->
            if (i in topKIndices) {
                logit
            } else {
                Value(Double.NEGATIVE_INFINITY)
            }
        }.toTypedArray()
    }

    private fun softmax(logits: Array<Value>): DoubleArray {
        // 수치 안정성을 위해 최대값 빼기
        val maxLogit = logits.maxByOrNull { it.data }?.data ?: 0.0
        val expValues = logits.map { exp(it.data - maxLogit) }
        val sum = expValues.sum()

        return expValues.map { it / sum }.toDoubleArray()
    }

    private fun sampleFromDistribution(probs: DoubleArray): Int {
        // 누적 분포 함수
        val cumsum = DoubleArray(probs.size)
        cumsum[0] = probs[0]
        for (i in 1 until probs.size) {
            cumsum[i] = cumsum[i - 1] + probs[i]
        }

        // 랜덤 샘플링
        val r = Random.nextDouble()
        for (i in cumsum.indices) {
            if (r <= cumsum[i]) {
                return i
            }
        }

        return probs.size - 1 // 안전장치
    }
}