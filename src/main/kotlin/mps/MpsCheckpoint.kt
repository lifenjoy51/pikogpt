package mps

import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import turbo.TurboModelConfig
import turbo.TurboTrainConfig
import java.io.File
import java.nio.ByteBuffer

/**
 * P0.1 + P3.2 — MPSGraph backend checkpoint metadata.
 *
 * 파일 layout (TurboCheckpoint와 schema 호환):
 *   checkpoint.json       — 이 클래스 직렬화 (TurboCheckpoint와 같은 field set)
 *   model_weights.bin     — params 순서대로 float32 BE concat (TurboTrainer.saveModelWeights와 동일)
 *   optimizer_state.bin   — Int(timeStep BE) + for p in params: m[..] then v[..] (TurboAdamW.saveState와 동일)
 *
 * **P3.2**: schema가 `turbo.TurboCheckpoint`와 호환되어 `TurboSampler`가 mps ckpt 디렉터리를
 * 그대로 로드해 sampling 가능 (params 순서/모델 구조 일치 시). `config` 필드는 turbo 호환을 위한
 * default `TurboTrainConfig` — mps 학습은 자체 `MpsGraphTrainConfig`를 사용하지만 ckpt schema는
 * turbo와 일치시킨다.
 */
@Serializable
data class MpsCheckpoint(
    val iterationNumber: Int,
    val bestValidationLoss: Double,
    val modelArgs: TurboModelConfig,
    val config: TurboTrainConfig = TurboTrainConfig(),
)

object MpsCheckpointIO {
    private val json = Json { prettyPrint = true; encodeDefaults = true; ignoreUnknownKeys = true }

    /**
     * session의 현재 weight + AdamW m/v를 host로 읽어 ckpt 디렉터리에 직렬화.
     * shapes는 paramIndex 순서대로의 weight shape 리스트 (session.loadWeights 호출 시 사용한 것 그대로).
     * timeStep은 mps step graph가 host에서 매 step 직접 주입하므로 ckpt에는 iterationNumber로 갈음.
     */
    fun save(
        session: MpsGraphSession,
        shapes: List<IntArray>,
        dir: File,
        meta: MpsCheckpoint,
    ) {
        dir.mkdirs()
        File(dir, "checkpoint.json").writeText(json.encodeToString(meta))
        saveModelWeights(session, shapes, File(dir, "model_weights.bin"))
        saveOptimizerState(session, shapes, meta.iterationNumber, File(dir, "optimizer_state.bin"))
    }

    /**
     * ckpt 디렉터리에서 weight + optimizer state를 session으로 로드.
     * session은 이미 create 된 상태여야 하며, 호출 후 loadWeights/loadOptimizerM/V가 paramIndex 순서대로 실행된다.
     */
    fun load(
        session: MpsGraphSession,
        shapes: List<IntArray>,
        dir: File,
    ): MpsCheckpoint {
        val meta = json.decodeFromString<MpsCheckpoint>(File(dir, "checkpoint.json").readText())
        loadModelWeights(session, shapes, File(dir, "model_weights.bin"))
        val optFile = File(dir, "optimizer_state.bin")
        if (optFile.exists()) {
            loadOptimizerState(session, shapes, optFile)
        }
        return meta
    }

    private fun saveModelWeights(session: MpsGraphSession, shapes: List<IntArray>, file: File) {
        file.outputStream().use { out ->
            for ((idx, shape) in shapes.withIndex()) {
                val numel = shape.fold(1) { a, b -> a * b }
                val data = FloatArray(numel)
                session.readWeight(idx, data)
                val buf = ByteBuffer.allocate(numel * 4)
                for (x in data) buf.putFloat(x)
                out.write(buf.array())
            }
        }
    }

    private fun saveOptimizerState(
        session: MpsGraphSession,
        shapes: List<IntArray>,
        timeStep: Int,
        file: File,
    ) {
        file.outputStream().use { out ->
            out.write(ByteBuffer.allocate(4).putInt(timeStep).array())
            for ((idx, _) in shapes.withIndex()) {
                val numel = shapes[idx].fold(1) { a, b -> a * b }
                val m = FloatArray(numel); session.readOptimizerM(idx, m)
                val v = FloatArray(numel); session.readOptimizerV(idx, v)
                val buf = ByteBuffer.allocate((m.size + v.size) * 4)
                for (x in m) buf.putFloat(x)
                for (x in v) buf.putFloat(x)
                out.write(buf.array())
            }
        }
    }

    private fun loadModelWeights(session: MpsGraphSession, shapes: List<IntArray>, file: File) {
        file.inputStream().use { input ->
            val word = ByteArray(4)
            for ((idx, shape) in shapes.withIndex()) {
                val numel = shape.fold(1) { a, b -> a * b }
                val data = FloatArray(numel)
                for (i in 0 until numel) {
                    require(input.read(word) == 4) { "weight EOF at idx=$idx i=$i" }
                    data[i] = ByteBuffer.wrap(word).float
                }
                session.loadWeights(idx, data, shape)
            }
        }
    }

    private fun loadOptimizerState(session: MpsGraphSession, shapes: List<IntArray>, file: File) {
        file.inputStream().use { input ->
            val hdr = ByteArray(4)
            require(input.read(hdr) == 4) { "optimizer_state.bin header EOF" }
            // timeStep header — 진입점 iterationNumber로 갈음하므로 여기서 사용 안 함.

            val word = ByteArray(4)
            for ((idx, shape) in shapes.withIndex()) {
                val numel = shape.fold(1) { a, b -> a * b }
                val m = FloatArray(numel)
                for (i in 0 until numel) {
                    require(input.read(word) == 4) { "m EOF at idx=$idx i=$i" }
                    m[i] = ByteBuffer.wrap(word).float
                }
                val v = FloatArray(numel)
                for (i in 0 until numel) {
                    require(input.read(word) == 4) { "v EOF at idx=$idx i=$i" }
                    v[i] = ByteBuffer.wrap(word).float
                }
                session.loadOptimizerM(idx, m)
                session.loadOptimizerV(idx, v)
            }
        }
    }
}
