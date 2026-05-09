package pipeline

import data.MetaInfo
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import sample.SampleConfig
import sample.ScalarSampler
import train.ScalarTrainer
import train.TrainConfig
import java.io.File
import java.nio.ByteBuffer
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * end-to-end 배관 검증 테스트.
 *
 * 합성 데이터 → Trainer 수 iter 학습 → 체크포인트 저장 → Sampler 로드 → 샘플링까지
 * 전체 흐름이 예외 없이 도는지 확인한다. 학습 품질은 검증하지 않는다.
 */
class FullPipelineTest {
    private val testRoot = File("build/tmp/pipeline-test")
    private val dataDir = File(testRoot, "data")
    private val modelDir = File(testRoot, "model")

    @AfterTest
    fun cleanup() {
        testRoot.deleteRecursively()
    }

    @Test
    fun trainThenSample() {
        prepareData()

        val trainConfig = TrainConfig(
            dataPath = dataDir.absolutePath,
            modelDir = modelDir.absolutePath,
            gradientAccumulationSteps = 1,
            batchSize = 1,
            blockSize = 8,
            numberOfLayers = 1,
            numberOfHeads = 1,
            embeddingDimension = 8,
            maxIters = 10,
            evalIntervalRatio = 0.1f,
            evalIters = 1,
            warmupRatio = 0.1f,
            learningRateDecayRatio = 1.0f,
            logInterval = 10,
            dropout = 0.0f,
        )
        ScalarTrainer(trainConfig).train()

        val checkpointDir = findCheckpointDir()
            ?: throw AssertionError("체크포인트 디렉토리가 생성되지 않음: ${modelDir.absolutePath}")
        assertTrue(File(checkpointDir, "checkpoint.json").exists(), "checkpoint.json 없음")
        assertTrue(File(checkpointDir, "model_weights.bin").exists(), "model_weights.bin 없음")
        assertTrue(File(checkpointDir, "meta.json").exists(), "meta.json 없음")

        val sampleConfig = SampleConfig(
            modelDirectoryPath = checkpointDir.absolutePath,
            numberOfSamples = 1,
            maximumNewTokens = 4,
            topKFilteringSize = 0,
            samplingTemperature = 1.0f,
        )
        val sampler = ScalarSampler(sampleConfig)
        val result = runBlocking { sampler.generateText("a") }

        assertTrue(result.results.isNotEmpty(), "Sampler 결과 없음")
        val generated = result.results[0]
        val vocabChars = setOf('a', 'b', 'c')
        assertTrue(
            generated.all { it in vocabChars },
            "생성된 토큰이 vocab({a,b,c}) 밖: '$generated'"
        )
    }

    private fun prepareData() {
        dataDir.mkdirs()
        modelDir.mkdirs()

        val meta = MetaInfo(
            vocabularySize = 4,
            indexToString = mapOf(0 to "<eos>", 1 to "a", 2 to "b", 3 to "c"),
            stringToIndex = mapOf("<eos>" to 0, "a" to 1, "b" to 2, "c" to 3)
        )
        File(dataDir, "meta.json").writeText(Json.encodeToString(meta))

        // "abc"를 반복한 토큰 시퀀스. DataLoader가 blockSize+1 이상을 요구하므로 넉넉히 생성.
        val trainTokens = List(40) { listOf(1, 2, 3) }.flatten()
        val valTokens = List(20) { listOf(1, 2, 3) }.flatten()
        writeBin(File(dataDir, "train.bin"), trainTokens)
        writeBin(File(dataDir, "val.bin"), valTokens)
    }

    private fun writeBin(file: File, tokens: List<Int>) {
        val buf = ByteBuffer.allocate(tokens.size * 4)
        tokens.forEach { buf.putInt(it) }
        file.writeBytes(buf.array())
    }

    private fun findCheckpointDir(): File? {
        // 새 schema: ${modelDir}/${datasetName}/${expName}/v0001/
        if (!modelDir.exists()) return null
        val datasetDir = modelDir.listFiles()?.firstOrNull { it.isDirectory } ?: return null
        val expDir = datasetDir.listFiles()?.firstOrNull { it.isDirectory } ?: return null
        return expDir.listFiles()?.firstOrNull {
            it.isDirectory && File(it, "checkpoint.json").exists()
        }
    }
}
