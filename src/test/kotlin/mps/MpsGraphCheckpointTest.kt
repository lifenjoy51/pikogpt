package mps

import kotlinx.serialization.json.Json
import org.junit.jupiter.api.Assumptions
import turbo.TurboCheckpoint
import java.io.File
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * P0.1 — Checkpoint save / load 회귀 보호.
 *
 * 1. m/v read/load roundtrip
 * 2. 학습 N step → save → 새 session에 load → 한 step 더 vs 직선 학습의 loss 일치 (rtol < 1e-3)
 * 3. checkpoint.json metadata roundtrip
 */
class MpsGraphCheckpointTest {

    private fun ensure() {
        val ok = MpsGraphSession.available()
        Assumptions.assumeTrue(ok, "MpsGraph unavailable: ${MpsGraphSession.loadError}")
    }

    private fun makeFixture(seed: Long = 31L): Fixture {
        val T = 8
        val embedDim = 16
        val numHeads = 4
        val numLayers = 1
        val vocab = 32
        val headDim = embedDim / numHeads
        val half = headDim / 2

        val gptCfg = gpt.GPTConfig(
            vocabularySize = vocab, embeddingDimension = embedDim,
            numberOfLayers = numLayers, numberOfAttentionHeads = numHeads,
            maxSequenceLength = T, dropoutProbability = 0.0f, useBias = true,
        )
        val modelCfg = turbo.TurboModelConfig(
            gpt = gptCfg, tieWeights = true, mlpActivation = "swiglu",
            positionEncoding = "rope", normalizationType = "layernorm",
        )
        val tm = turbo.layer.TurboPikoGPT(modelCfg)
        val rng = Random(seed)
        for (p in tm.parameters()) for (i in 0 until p.numel) p.data[i] = (rng.nextFloat() - 0.5f) * 0.2f

        val cos = FloatArray(T * half)
        val sin = FloatArray(T * half)
        for (t in 0 until T) for (i in 0 until half) {
            val theta = Math.pow(10000.0, -2.0 * i / headDim)
            cos[t * half + i] = kotlin.math.cos(t * theta).toFloat()
            sin[t * half + i] = kotlin.math.sin(t * theta).toFloat()
        }
        val mask = FloatArray(T * T) { idx -> if (idx % T > idx / T) -1e9f else 0f }
        val tokens = IntArray(T) { rng.nextInt(vocab) }
        val targets = IntArray(T) { rng.nextInt(vocab) }
        val cfg = MpsGraphConfig(numLayers, embedDim, numHeads, T, vocab, batchSize = 1)
        val shapes = tm.parameters().map { it.shape }
        return Fixture(cfg, modelCfg, tm, shapes, tokens, targets, cos, sin, mask)
    }

    @Test
    fun optimizerMVRoundTrip() {
        ensure()
        val fx = makeFixture()
        MpsGraphSession.create(fx.cfg).use { s ->
            for ((idx, p) in fx.tm.parameters().withIndex()) {
                s.loadWeights(idx, p.data, p.shape)
            }
            // 첫 step 돌려 m/v를 0이 아닌 값으로 만든다.
            s.runTrainingStep(fx.tokens, fx.targets, fx.cos, fx.sin, fx.mask,
                lr = 1e-2f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f,
                weightDecay = 0.0f, stepT = 1)

            // 첫 paramIndex만 검사. m/v read → load → read 후 동일.
            val idx = 0
            val numel = fx.shapes[idx].fold(1) { a, b -> a * b }
            val m1 = FloatArray(numel); s.readOptimizerM(idx, m1)
            val v1 = FloatArray(numel); s.readOptimizerV(idx, v1)

            // m/v를 임의 값으로 덮어쓰고 다시 read해서 같은지 확인.
            val mNew = FloatArray(numel) { it * 0.1f }
            val vNew = FloatArray(numel) { 1.0f + it * 0.01f }
            s.loadOptimizerM(idx, mNew)
            s.loadOptimizerV(idx, vNew)
            val m2 = FloatArray(numel); s.readOptimizerM(idx, m2)
            val v2 = FloatArray(numel); s.readOptimizerV(idx, v2)
            for (i in 0 until numel) {
                assertEquals(mNew[i], m2[i], 1e-7f, "m[$i] roundtrip 불일치")
                assertEquals(vNew[i], v2[i], 1e-7f, "v[$i] roundtrip 불일치")
            }
            // 원래 값과는 달라야 한다 (덮어쓰기 확인).
            val diff = (0 until numel).sumOf { abs(m1[it] - m2[it]).toDouble() }
            assertTrue(diff > 0.0, "load가 실제로 덮어쓰지 않음")
        }
    }

    @Test
    fun saveLoadRestoresExactState() {
        ensure()
        val fx = makeFixture(seed = 41L)
        val tmpDir = File.createTempFile("mps-ckpt-test-", "-dir").also {
            it.delete(); it.mkdirs()
        }
        try {
            // 시나리오 A: 5 step → 6번째 step의 loss 기록
            val lrParams = AdamParams(lr = 1e-2f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f, wd = 0.0f)
            val lossA: Float
            MpsGraphSession.create(fx.cfg).use { s ->
                for ((idx, p) in fx.tm.parameters().withIndex()) {
                    s.loadWeights(idx, p.data, p.shape)
                }
                for (step in 1..5) {
                    s.runTrainingStep(fx.tokens, fx.targets, fx.cos, fx.sin, fx.mask,
                        lrParams.lr, lrParams.beta1, lrParams.beta2, lrParams.eps, lrParams.wd, step)
                }
                lossA = s.runTrainingStep(fx.tokens, fx.targets, fx.cos, fx.sin, fx.mask,
                    lrParams.lr, lrParams.beta1, lrParams.beta2, lrParams.eps, lrParams.wd, 6)
            }

            // 시나리오 B: 5 step → save → 새 session에 load → 6번째 step의 loss 기록
            val lossB: Float
            val meta = MpsCheckpoint(
                iterationNumber = 5,
                bestValidationLoss = -1.0,
                modelArgs = fx.modelCfg,
            )
            MpsGraphSession.create(fx.cfg).use { s ->
                for ((idx, p) in fx.tm.parameters().withIndex()) {
                    s.loadWeights(idx, p.data, p.shape)
                }
                for (step in 1..5) {
                    s.runTrainingStep(fx.tokens, fx.targets, fx.cos, fx.sin, fx.mask,
                        lrParams.lr, lrParams.beta1, lrParams.beta2, lrParams.eps, lrParams.wd, step)
                }
                MpsCheckpointIO.save(s, fx.shapes, tmpDir, meta)
            }
            MpsGraphSession.create(fx.cfg).use { s2 ->
                val loaded = MpsCheckpointIO.load(s2, fx.shapes, tmpDir)
                assertEquals(5, loaded.iterationNumber)
                lossB = s2.runTrainingStep(fx.tokens, fx.targets, fx.cos, fx.sin, fx.mask,
                    lrParams.lr, lrParams.beta1, lrParams.beta2, lrParams.eps, lrParams.wd, 6)
            }

            val rdiff = abs(lossA - lossB) / abs(lossA).coerceAtLeast(1e-6f)
            assertTrue(rdiff < 1e-3f,
                "직선 학습 vs save/load 후 학습의 6번째 step loss 불일치: A=$lossA B=$lossB rdiff=$rdiff")
        } finally {
            tmpDir.deleteRecursively()
        }
    }

    /**
     * P3.2 옵션 A — mps ckpt의 checkpoint.json schema가 TurboCheckpoint와 호환되어
     * TurboSampler가 그대로 로드 가능. 여기선 schema deserialize만 검증 (실제 sampling은
     * trainer 외부에서 ./gradlew runSamplePromptsFromFile 등으로 실행 가능).
     */
    @Test
    fun ckptSchemaIsTurboCompatible() {
        ensure()
        val fx = makeFixture(seed = 61L)
        val tmpDir = File.createTempFile("mps-ckpt-turbo-", "-dir").also {
            it.delete(); it.mkdirs()
        }
        try {
            val meta = MpsCheckpoint(
                iterationNumber = 42,
                bestValidationLoss = 3.14,
                modelArgs = fx.modelCfg,
            )
            MpsGraphSession.create(fx.cfg).use { s ->
                for ((idx, p) in fx.tm.parameters().withIndex()) {
                    s.loadWeights(idx, p.data, p.shape)
                }
                MpsCheckpointIO.save(s, fx.shapes, tmpDir, meta)
            }

            val parser = Json { ignoreUnknownKeys = true }
            val turboMeta = parser.decodeFromString<TurboCheckpoint>(
                File(tmpDir, "checkpoint.json").readText()
            )
            assertEquals(42, turboMeta.iterationNumber)
            assertEquals(3.14, turboMeta.bestValidationLoss, 1e-9)
            assertEquals(fx.modelCfg.gpt.vocabularySize, turboMeta.modelArgs.gpt.vocabularySize)
            assertEquals(fx.modelCfg.tieWeights, turboMeta.modelArgs.tieWeights)
        } finally {
            tmpDir.deleteRecursively()
        }
    }

    @Test
    fun checkpointJsonRoundTrip() {
        ensure()
        val fx = makeFixture(seed = 51L)
        val tmpDir = File.createTempFile("mps-ckpt-meta-", "-dir").also {
            it.delete(); it.mkdirs()
        }
        try {
            val meta = MpsCheckpoint(
                iterationNumber = 123,
                bestValidationLoss = 2.345,
                modelArgs = fx.modelCfg,
            )
            MpsGraphSession.create(fx.cfg).use { s ->
                for ((idx, p) in fx.tm.parameters().withIndex()) {
                    s.loadWeights(idx, p.data, p.shape)
                }
                MpsCheckpointIO.save(s, fx.shapes, tmpDir, meta)
            }
            MpsGraphSession.create(fx.cfg).use { s2 ->
                for ((idx, p) in fx.tm.parameters().withIndex()) {
                    s2.loadWeights(idx, p.data, p.shape)
                }
                val loaded = MpsCheckpointIO.load(s2, fx.shapes, tmpDir)
                assertEquals(123, loaded.iterationNumber)
                assertEquals(2.345, loaded.bestValidationLoss, 1e-9)
                assertEquals(fx.modelCfg.gpt.vocabularySize, loaded.modelArgs.gpt.vocabularySize)
                assertEquals(fx.modelCfg.tieWeights, loaded.modelArgs.tieWeights)
            }
        } finally {
            tmpDir.deleteRecursively()
        }
    }

    private data class Fixture(
        val cfg: MpsGraphConfig,
        val modelCfg: turbo.TurboModelConfig,
        val tm: turbo.layer.TurboPikoGPT,
        val shapes: List<IntArray>,
        val tokens: IntArray,
        val targets: IntArray,
        val cos: FloatArray,
        val sin: FloatArray,
        val mask: FloatArray,
    )

    private data class AdamParams(
        val lr: Float, val beta1: Float, val beta2: Float, val eps: Float, val wd: Float,
    )
}
