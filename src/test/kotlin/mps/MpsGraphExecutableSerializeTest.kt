package mps

import org.junit.jupiter.api.Assumptions
import java.io.File
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * P3.1 — MPSGraphExecutable compile + serialize / deserialize 검증.
 *
 *   1) session에서 stepGraph compile → MPSGraphPackage로 serialize
 *   2) 다른 session에서 같은 package deserialize → 성공
 *
 * run path는 본 PoC에서 graph 기반 유지 (executable inputsArray ordering refactor가 별도 작업).
 * 본 test는 API 도입 자체 + 디스크 roundtrip 검증.
 */
class MpsGraphExecutableSerializeTest {

    private fun ensure() {
        Assumptions.assumeTrue(MpsGraphSession.available(),
            "MpsGraph unavailable: ${MpsGraphSession.loadError}")
    }

    private fun mkCfg(): MpsGraphConfig = MpsGraphConfig(
        numLayers = 1, embedDim = 16, numHeads = 4, blockSize = 8,
        vocab = 32, batchSize = 1,
    )

    private fun mkModel(cfg: MpsGraphConfig): turbo.layer.TurboPikoGPT {
        val gptCfg = gpt.GPTConfig(
            vocabularySize = cfg.vocab, embeddingDimension = cfg.embedDim,
            numberOfLayers = cfg.numLayers, numberOfAttentionHeads = cfg.numHeads,
            maxSequenceLength = cfg.blockSize, dropoutProbability = 0.0f, useBias = true,
        )
        val modelCfg = turbo.TurboModelConfig(
            gpt = gptCfg, tieWeights = true, mlpActivation = "swiglu",
            positionEncoding = "rope", normalizationType = "layernorm",
        )
        return turbo.layer.TurboPikoGPT(modelCfg)
    }

    @Test
    fun compileAndSerializeRoundtrip() {
        ensure()
        val cfg = mkCfg()
        val tmpDir = Files.createTempDirectory("mps_pkg_").toFile()
        val pkgDir = File(tmpDir, "step.mpsgraphpackage")

        val tm = mkModel(cfg)
        MpsGraphSession.create(cfg).use { s ->
            for ((idx, p) in tm.parameters().withIndex()) s.loadWeights(idx, p.data, p.shape)
            val ok = s.compileStepAndSerialize(
                pkgDir.absolutePath, batchSize = cfg.batchSize, blockSize = cfg.blockSize,
            )
            assertTrue(ok, "compileStepAndSerialize failed")
            assertTrue(pkgDir.exists(), "package dir not created: ${pkgDir.absolutePath}")
        }

        // 새 session에서 deserialize 시도
        MpsGraphSession.create(cfg).use { s2 ->
            for ((idx, p) in tm.parameters().withIndex()) s2.loadWeights(idx, p.data, p.shape)
            val loaded = s2.loadStepExecutable(pkgDir.absolutePath)
            assertTrue(loaded, "loadStepExecutable failed for ${pkgDir.absolutePath}")
        }

        // cleanup
        pkgDir.deleteRecursively()
        tmpDir.deleteRecursively()
    }

    @Test
    fun loadNonexistentReturnsFalse() {
        ensure()
        val cfg = mkCfg()
        MpsGraphSession.create(cfg).use { s ->
            val tm = mkModel(cfg)
            for ((idx, p) in tm.parameters().withIndex()) s.loadWeights(idx, p.data, p.shape)
            val ok = s.loadStepExecutable("/tmp/__nonexistent_mpsgraph_package_xyz/")
            assertTrue(!ok, "nonexistent path가 false 반환해야: got $ok")
        }
    }
}
