package vec

import data.MetaInfo
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import sample.SampleConfig
import train.TrainConfig
import java.io.File
import java.nio.ByteBuffer
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * 벡터 백엔드의 end-to-end 파이프라인 테스트.
 *
 * 합성 vocab(a/b/c)으로 몇 iter 학습 → 체크포인트 저장 → Sampler 로드 → 생성까지가
 * 예외 없이 돌아가는지 확인. 학습 품질은 검증하지 않음. 1초 이내 목표.
 */
class VFullPipelineTest {
    private val testRoot = File("build/tmp/vec-pipeline-test")
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
            maxIters = 5,
            evalIntervalRatio = 0.2f,
            evalIters = 1,
            warmupRatio = 0.1f,
            learningRateDecayRatio = 1.0f,
            logInterval = 10,
            dropout = 0.0f,
        )
        VecTrainer(trainConfig).train()

        // 체크포인트 찾기 — 벡터 루트는 modelDir/{datasetName}/vec/{params}/{loss*10}/
        // datasetName = dataPath 마지막 segment (이 테스트에선 tmp "data" dir).
        val datasetDir = File(modelDir, dataDir.name)
        val vecRoot = File(datasetDir, "vec")
        val paramDir = vecRoot.listFiles()?.firstOrNull { it.isDirectory }
            ?: throw AssertionError("vec 파라미터 디렉토리 없음: ${vecRoot.absolutePath}")
        val ckpt = paramDir.listFiles()?.firstOrNull {
            it.isDirectory && File(it, "checkpoint.json").exists()
        } ?: throw AssertionError("체크포인트 디렉토리 없음: ${paramDir.absolutePath}")

        assertTrue(File(ckpt, "checkpoint.json").exists())
        assertTrue(File(ckpt, "model_weights.bin").exists())
        assertTrue(File(ckpt, "meta.json").exists())

        val sampleConfig = SampleConfig(
            modelDirectoryPath = ckpt.absolutePath,
            numberOfSamples = 1,
            maximumNewTokens = 4,
            samplingTemperature = 1.0f,
            topKFilteringSize = 0,
        )
        val sampler = VecSampler(sampleConfig)
        val outputs = sampler.generate("a")

        assertTrue(outputs.isNotEmpty(), "Sampler 생성 결과가 비어 있음")
        val vocabChars = setOf('a', 'b', 'c')
        assertTrue(
            outputs[0].all { it in vocabChars },
            "생성된 문자가 vocab({a,b,c}) 밖: '${outputs[0]}'"
        )
    }

    private fun prepareData() {
        dataDir.mkdirs()
        modelDir.mkdirs()

        val meta = MetaInfo(
            vocabularySize = 4,
            indexToString = mapOf(0 to "<|eos|>", 1 to "a", 2 to "b", 3 to "c"),
            stringToIndex = mapOf("<|eos|>" to 0, "a" to 1, "b" to 2, "c" to 3),
        )
        File(dataDir, "meta.json").writeText(Json.encodeToString(meta))

        val trainTokens = List(40) { listOf(1, 2, 3) }.flatten()
        val valTokens = List(20) { listOf(1, 2, 3) }.flatten()
        writeBin(File(dataDir, "train.bin"), trainTokens)
        writeBin(File(dataDir, "val.bin"), valTokens)
    }

    private fun writeBin(file: File, tokens: List<Int>) {
        val buf = ByteBuffer.allocate(tokens.size * 4)
        for (t in tokens) buf.putInt(t)
        file.writeBytes(buf.array())
    }
}
