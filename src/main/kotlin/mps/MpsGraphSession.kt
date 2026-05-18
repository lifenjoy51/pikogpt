package mps

import java.io.File

/**
 * MPSGraph 기반 GPU 100% residence 학습 backend의 JNI 진입점.
 *
 * Phase 1: 인프라 — `nativeInit`/`nativeCreateSession`/`nativeDestroySession`만 구현.
 * Phase 2+ 에서 weight load / forward / backward / AdamW step 채움.
 *
 * 사용 예 (Phase 5+):
 * ```
 * val session = MpsGraphSession.create(config)
 * session.use { s ->
 *     s.loadWeights(...)
 *     repeat(iters) { s.step(tokenIds, targets) }
 * }
 * ```
 */
class MpsGraphSession private constructor(handle: Long) : AutoCloseable {

    @Volatile private var handle: Long = handle

    fun loadWeights(paramIndex: Int, data: FloatArray, shape: IntArray) {
        check(handle != 0L) { "session closed" }
        val numel = shape.fold(1) { acc, d -> acc * d }
        require(numel == data.size) {
            "data size ${data.size} != product(shape) $numel (shape=${shape.toList()})"
        }
        nativeLoadWeights(handle, paramIndex, data, shape)
    }

    fun weightCount(): Int {
        check(handle != 0L) { "session closed" }
        return nativeWeightCount(handle)
    }

    /**
     * Phase 2.2 step 1: tokenEmbedding(paramIndex 0) lookup만. tokenIds → [tokLen, embedDim].
     *
     * Phase 2.3+ 에서 LayerNorm/MHA/SwiGLU/lm_head 추가하며 graph 확장하고 이 함수는 제거.
     */
    fun runEmbeddingForward(tokenIds: IntArray, output: FloatArray) {
        check(handle != 0L) { "session closed" }
        nativeRunEmbeddingForward(handle, tokenIds, output)
    }

    /**
     * Phase 4+5: forward + loss + backward + AdamW. 1 GPU dispatch = 1 training step.
     * weight/m/v는 GPU resident (slot에 보관). step 호출 후 자동 갱신.
     */
    fun runTrainingStep(
        tokenIds: IntArray, targets: IntArray,
        cos: FloatArray, sin: FloatArray, mask: FloatArray,
        lr: Float, beta1: Float, beta2: Float, eps: Float, weightDecay: Float,
        stepT: Int,
        gradClip: Float = 0.0f,
        batchSize: Int = 1,
        dropoutMask: FloatArray? = null,
    ): Float {
        check(handle != 0L) { "session closed" }
        require(tokenIds.size == targets.size)
        require(tokenIds.size % batchSize == 0) {
            "tokenIds.size ${tokenIds.size} not divisible by batchSize $batchSize"
        }
        return nativeRunTrainingStep(handle, tokenIds, targets, cos, sin, mask,
            lr, beta1, beta2, eps, weightDecay, gradClip, stepT, batchSize, dropoutMask)
    }

    /** 검증/체크포인트용 weight 읽기. */
    fun readWeight(paramIndex: Int, out: FloatArray) {
        check(handle != 0L) { "session closed" }
        nativeReadWeight(handle, paramIndex, out)
    }

    /**
     * P1.2 — 1 micro-step accumulate. forward + loss + backward 후 grad를 slot.gradBuffer에 누적.
     * loss return. AdamW는 적용 안 됨 — N micro-step 후 runAdamStep 호출.
     * gradient는 자동 누적 (accumGraph 안에서 gradOld + new_grad).
     * batchSize 인자: tokenIds.size = batchSize * blockSize여야 한다.
     */
    fun runAccumStep(
        tokenIds: IntArray, targets: IntArray,
        cos: FloatArray, sin: FloatArray, mask: FloatArray,
        batchSize: Int = 1,
        dropoutMask: FloatArray? = null,
    ): Float {
        check(handle != 0L) { "session closed" }
        require(tokenIds.size == targets.size)
        require(tokenIds.size % batchSize == 0) {
            "tokenIds.size ${tokenIds.size} not divisible by batchSize $batchSize"
        }
        return nativeRunAccumStep(handle, tokenIds, targets, cos, sin, mask, batchSize, dropoutMask)
    }

    /**
     * Phase 3 — fused step: accumSteps × forward+backward+grad accumulate + AdamW를 단일
     * MTLCommandBuffer에 encode + commit + wait. iter당 9번 분리 GPU run → 1번 commit으로 묶음.
     *
     * 입력 모두 flatten:
     *   allTokenIds : [accumSteps * batchSize * blockSize]
     *   allTargets  : [accumSteps * batchSize * blockSize]
     *   allDropoutMask : [accumSteps * 2*L*B*T*E] (useDropout=true 시점만 사용)
     *
     * return: 마지막 micro-step loss.
     */
    fun runFusedStep(
        allTokenIds: IntArray, allTargets: IntArray,
        cos: FloatArray, sin: FloatArray, mask: FloatArray,
        allDropoutMask: FloatArray?,
        lr: Float, beta1: Float, beta2: Float, eps: Float, weightDecay: Float,
        gradClip: Float, stepT: Int, batchSize: Int, accumSteps: Int,
    ): Float {
        check(handle != 0L) { "session closed" }
        require(allTokenIds.size == allTargets.size)
        require(accumSteps >= 1)
        require(batchSize >= 1)
        require(allTokenIds.size % (accumSteps * batchSize) == 0)
        return nativeRunFusedStep(handle, allTokenIds, allTargets, cos, sin, mask, allDropoutMask,
            lr, beta1, beta2, eps, weightDecay, gradClip, stepT, batchSize, accumSteps)
    }

    /**
     * P1.2 — AdamW 1 step. slot.gradBuffer (누적된 grad) 사용 + reset.
     * runAccumStep N번 → runAdamStep 1번 호출이 1 effective iter.
     */
    fun runAdamStep(
        lr: Float, beta1: Float, beta2: Float, eps: Float, weightDecay: Float,
        gradClip: Float, stepT: Int,
    ) {
        check(handle != 0L) { "session closed" }
        nativeRunAdamStep(handle, lr, beta1, beta2, eps, weightDecay, gradClip, stepT)
    }

    /** P1.2 — 모든 slot.gradBuffer를 0으로 reset. 학습 시작 시 호출. */
    fun resetGradAccum() {
        check(handle != 0L) { "session closed" }
        nativeResetGradAccum(handle)
    }

    /** P0.1 — AdamW m state 읽기 (체크포인트용). */
    fun readOptimizerM(paramIndex: Int, out: FloatArray) {
        check(handle != 0L) { "session closed" }
        nativeReadOptimizerM(handle, paramIndex, out)
    }

    /** P0.1 — AdamW v state 읽기 (체크포인트용). */
    fun readOptimizerV(paramIndex: Int, out: FloatArray) {
        check(handle != 0L) { "session closed" }
        nativeReadOptimizerV(handle, paramIndex, out)
    }

    /** P0.1 — AdamW m state 덮어쓰기 (resume용). */
    fun loadOptimizerM(paramIndex: Int, data: FloatArray) {
        check(handle != 0L) { "session closed" }
        nativeLoadOptimizerM(handle, paramIndex, data)
    }

    /** P0.1 — AdamW v state 덮어쓰기 (resume용). */
    fun loadOptimizerV(paramIndex: Int, data: FloatArray) {
        check(handle != 0L) { "session closed" }
        nativeLoadOptimizerV(handle, paramIndex, data)
    }

    /**
     * Phase 3 step 2: forward + CE loss + backward(gradient). 모든 weight gradient를 GPU buffer (slot.gradBuffer)에 저장.
     * 다음 step (AdamW)가 같은 buffer를 input으로 사용. caller는 loss만 받음.
     */
    fun runForwardBackward(
        tokenIds: IntArray, targets: IntArray,
        cos: FloatArray, sin: FloatArray, mask: FloatArray,
    ): Float {
        check(handle != 0L) { "session closed" }
        require(tokenIds.size == targets.size)
        return nativeRunForwardBackward(handle, tokenIds, targets, cos, sin, mask)
    }

    /** Phase 3 검증용: GPU grad buffer를 host로 복사. */
    fun readGrad(paramIndex: Int, out: FloatArray) {
        check(handle != 0L) { "session closed" }
        nativeReadGrad(handle, paramIndex, out)
    }

    /**
     * Phase 3 step 1: forward + CE loss. scalar loss return.
     * targets: next-token labels per position. tokenIds.size = batchSize * blockSize.
     * P1.1 — B>1 일반화. batchSize default 1로 backward compatible.
     */
    fun runForwardLoss(
        tokenIds: IntArray, targets: IntArray,
        cos: FloatArray, sin: FloatArray, mask: FloatArray,
        batchSize: Int = 1,
    ): Float {
        check(handle != 0L) { "session closed" }
        require(tokenIds.size == targets.size) { "tokens.size != targets.size" }
        require(tokenIds.size % batchSize == 0) {
            "tokenIds.size ${tokenIds.size} not divisible by batchSize $batchSize"
        }
        return nativeRunForwardLoss(handle, tokenIds, targets, cos, sin, mask, batchSize)
    }

    /**
     * P3.1 — stepGraph를 compile → MPSGraphPackage 디렉터리로 serialize.
     * 디스크에 저장된 package는 같은 모델/dtype/shape configuration의 다음 session에서 deserialize 가능.
     * @return 성공 여부.
     */
    fun compileStepAndSerialize(
        path: String, batchSize: Int, blockSize: Int,
        beta1: Float = 0.9f, beta2: Float = 0.95f, eps: Float = 1e-8f, weightDecay: Float = 0.0f,
    ): Boolean {
        check(handle != 0L) { "session closed" }
        return nativeCompileStepAndSerialize(handle, path, batchSize, blockSize, beta1, beta2, eps, weightDecay)
    }

    /**
     * P3.1 — 디스크의 MPSGraphPackage를 load. 성공 시 cold start 절감 가능.
     * 본 PoC는 deserialize 검증만 (run path는 graph 기반 유지).
     */
    fun loadStepExecutable(path: String): Boolean {
        check(handle != 0L) { "session closed" }
        return nativeLoadStepExecutable(handle, path)
    }

    /**
     * Phase 2.2 step 6+7: Full forward (numLayers block + tied lm head) — GPU 100% path.
     * tokens [T] → logits [T, vocab]. cos/sin/mask는 caller가 미리 계산.
     */
    fun runFullForward(
        tokenIds: IntArray, cos: FloatArray, sin: FloatArray, mask: FloatArray,
        logits: FloatArray,
    ) {
        check(handle != 0L) { "session closed" }
        nativeRunFullForward(handle, tokenIds, cos, sin, mask, logits)
    }

    /**
     * Phase 2.2 step 5: Multi-head causal self-attention + RoPE.
     * cos/sin/mask는 caller가 [T, headDim/2], [T, T]로 미리 계산.
     */
    fun runAttentionForward(
        pQW: Int, pQB: Int, pKW: Int, pKB: Int, pVW: Int, pVB: Int,
        pOutW: Int, pOutB: Int,
        T: Int, embedDim: Int, numHeads: Int,
        input: FloatArray, cos: FloatArray, sin: FloatArray, mask: FloatArray,
        output: FloatArray,
    ) {
        check(handle != 0L) { "session closed" }
        nativeRunAttentionForward(handle, pQW, pQB, pKW, pKB, pVW, pVB, pOutW, pOutB,
            T, embedDim, numHeads, input, cos, sin, mask, output)
    }

    /**
     * Phase 2.2 step 4: SwiGLU MLP forward 단독.
     *   gate = x @ gateW.T + gateB
     *   up   = x @ upW.T + upB
     *   h    = silu(gate) * up
     *   y    = h @ downW.T + downB
     */
    fun runSwiGluForward(
        pGateW: Int, pGateB: Int, pUpW: Int, pUpB: Int, pDownW: Int, pDownB: Int,
        T: Int, embedDim: Int, hiddenDim: Int,
        input: FloatArray, output: FloatArray,
    ) {
        check(handle != 0L) { "session closed" }
        require(input.size == T * embedDim && output.size == T * embedDim)
        nativeRunSwiGluForward(handle, pGateW, pGateB, pUpW, pUpB, pDownW, pDownB,
            T, embedDim, hiddenDim, input, output)
    }

    /** Phase 2.2 step 3: Linear forward 단독. y = x @ weight.T + bias. paramBias=-1이면 bias 없음. */
    fun runLinearForward(
        paramWeight: Int, paramBias: Int,
        T: Int, inF: Int, outF: Int,
        input: FloatArray, output: FloatArray,
    ) {
        check(handle != 0L) { "session closed" }
        require(input.size == T * inF && output.size == T * outF)
        nativeRunLinearForward(handle, paramWeight, paramBias, T, inF, outF, input, output)
    }

    /** Phase 2.2 step 2: LayerNorm forward 단독. shape: input/output [T, C]. */
    fun runLayerNormForward(
        paramGamma: Int, paramBeta: Int,
        T: Int, C: Int, eps: Float,
        input: FloatArray, output: FloatArray,
    ) {
        check(handle != 0L) { "session closed" }
        require(input.size == T * C && output.size == T * C)
        nativeRunLayerNormForward(handle, paramGamma, paramBeta, T, C, eps, input, output)
    }

    /** Phase 5: 1 step = forward + backward + AdamW (GPU). loss return. */
    fun step(tokenIds: IntArray, targets: IntArray): Float {
        check(handle != 0L) { "session closed" }
        return nativeStep(handle, tokenIds, targets)
    }

    @Synchronized
    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            nativeDestroySession(h)
        }
    }

    companion object {
        @Volatile private var libLoaded: Boolean = false
        @Volatile var loadError: String? = null
            private set

        /**
         * MPSGraph 사용 가능 여부. macOS arm64 + libpikogpt_mpsgraph.dylib + MPSGraph init OK.
         * 한 번 평가 후 캐시.
         */
        @Synchronized
        fun available(): Boolean {
            if (libLoaded) return true
            val osName = System.getProperty("os.name").lowercase()
            val osArch = System.getProperty("os.arch").lowercase()
            if (!osName.contains("mac")) {
                loadError = "not macOS"
                return false
            }
            if (osArch != "aarch64" && osArch != "arm64") {
                loadError = "not arm64"
                return false
            }
            val dylib = resolveDylib() ?: run {
                loadError = "libpikogpt_mpsgraph.dylib not found (run ./gradlew buildMpsGraphLib)"
                return false
            }
            try {
                System.load(dylib.absolutePath)
            } catch (t: Throwable) {
                loadError = "System.load failed: ${t.message}"
                return false
            }
            val ok = try {
                nativeInit()
            } catch (t: Throwable) {
                loadError = "nativeInit threw: ${t.message}"
                return false
            }
            if (!ok) {
                loadError = "nativeInit returned false"
                return false
            }
            libLoaded = true
            loadError = null
            return true
        }

        fun create(config: MpsGraphConfig): MpsGraphSession {
            check(available()) { "MpsGraphSession unavailable: $loadError" }
            val h = nativeCreateSession(
                config.numLayers,
                config.embedDim,
                config.numHeads,
                config.blockSize,
                config.vocab,
                config.batchSize,
                config.useSwiglu,
                config.useRope,
                config.tieWeights,
                config.useVariableForStep,
                config.useFp16,
                config.useDropout,
                config.dropoutProbability,
                config.gradientAccumulationSteps,
            )
            check(h != 0L) { "nativeCreateSession returned 0" }
            return MpsGraphSession(h)
        }

        private fun resolveDylib(): File? {
            System.getProperty("java.library.path")?.split(File.pathSeparator)?.forEach { dir ->
                val f = File(dir, "libpikogpt_mpsgraph.dylib")
                if (f.exists()) return f
            }
            val cwd = File("build/native/libpikogpt_mpsgraph.dylib")
            if (cwd.exists()) return cwd
            return null
        }

        @JvmStatic private external fun nativeInit(): Boolean
        @JvmStatic private external fun nativeCreateSession(
            numLayers: Int, embedDim: Int, numHeads: Int, blockSize: Int,
            vocab: Int, batchSize: Int,
            useSwiglu: Boolean, useRope: Boolean, tieWeights: Boolean,
            useVariableForStep: Boolean,
            useFp16: Boolean,
            useDropout: Boolean,
            dropoutProbability: Float,
            gradientAccumulationSteps: Int,
        ): Long
        @JvmStatic private external fun nativeDestroySession(handle: Long)
        @JvmStatic private external fun nativeLoadWeights(handle: Long, paramIndex: Int, data: FloatArray, shape: IntArray)
        @JvmStatic private external fun nativeWeightCount(handle: Long): Int
        @JvmStatic private external fun nativeRunEmbeddingForward(
            handle: Long, tokenIds: IntArray, output: FloatArray)
        @JvmStatic private external fun nativeRunForwardLoss(
            handle: Long,
            tokenIds: IntArray, targets: IntArray,
            cos: FloatArray, sin: FloatArray, mask: FloatArray,
            batchSize: Int): Float
        // P3.1 — MPSGraphExecutable compile + serialize / deserialize
        @JvmStatic private external fun nativeCompileStepAndSerialize(
            handle: Long, path: String,
            batchSize: Int, blockSize: Int,
            beta1: Float, beta2: Float, eps: Float, weightDecay: Float): Boolean
        @JvmStatic private external fun nativeLoadStepExecutable(
            handle: Long, path: String): Boolean
        @JvmStatic private external fun nativeRunForwardBackward(
            handle: Long,
            tokenIds: IntArray, targets: IntArray,
            cos: FloatArray, sin: FloatArray, mask: FloatArray): Float
        @JvmStatic private external fun nativeReadGrad(
            handle: Long, paramIndex: Int, out: FloatArray)
        @JvmStatic private external fun nativeRunTrainingStep(
            handle: Long,
            tokenIds: IntArray, targets: IntArray,
            cos: FloatArray, sin: FloatArray, mask: FloatArray,
            lr: Float, beta1: Float, beta2: Float, eps: Float, weightDecay: Float,
            gradClip: Float, stepT: Int, batchSize: Int,
            dropoutMask: FloatArray?): Float
        @JvmStatic private external fun nativeReadWeight(
            handle: Long, paramIndex: Int, out: FloatArray)
        @JvmStatic private external fun nativeRunAccumStep(
            handle: Long,
            tokenIds: IntArray, targets: IntArray,
            cos: FloatArray, sin: FloatArray, mask: FloatArray,
            batchSize: Int,
            dropoutMask: FloatArray?): Float
        @JvmStatic private external fun nativeRunFusedStep(
            handle: Long,
            allTokenIds: IntArray, allTargets: IntArray,
            cos: FloatArray, sin: FloatArray, mask: FloatArray,
            allDropoutMask: FloatArray?,
            lr: Float, beta1: Float, beta2: Float, eps: Float, weightDecay: Float,
            gradClip: Float, stepT: Int, batchSize: Int, accumSteps: Int): Float
        @JvmStatic private external fun nativeRunAdamStep(
            handle: Long,
            lr: Float, beta1: Float, beta2: Float, eps: Float, weightDecay: Float,
            gradClip: Float, stepT: Int)
        @JvmStatic private external fun nativeResetGradAccum(handle: Long)
        @JvmStatic private external fun nativeReadOptimizerM(
            handle: Long, paramIndex: Int, out: FloatArray)
        @JvmStatic private external fun nativeReadOptimizerV(
            handle: Long, paramIndex: Int, out: FloatArray)
        @JvmStatic private external fun nativeLoadOptimizerM(
            handle: Long, paramIndex: Int, data: FloatArray)
        @JvmStatic private external fun nativeLoadOptimizerV(
            handle: Long, paramIndex: Int, data: FloatArray)
        @JvmStatic private external fun nativeRunFullForward(
            handle: Long,
            tokenIds: IntArray, cos: FloatArray, sin: FloatArray, mask: FloatArray,
            logits: FloatArray)
        @JvmStatic private external fun nativeRunAttentionForward(
            handle: Long,
            pQW: Int, pQB: Int, pKW: Int, pKB: Int, pVW: Int, pVB: Int,
            pOutW: Int, pOutB: Int,
            T: Int, embedDim: Int, numHeads: Int,
            input: FloatArray, cos: FloatArray, sin: FloatArray, mask: FloatArray,
            output: FloatArray)
        @JvmStatic private external fun nativeRunSwiGluForward(
            handle: Long,
            pGateW: Int, pGateB: Int, pUpW: Int, pUpB: Int, pDownW: Int, pDownB: Int,
            T: Int, embedDim: Int, hiddenDim: Int,
            input: FloatArray, output: FloatArray)
        @JvmStatic private external fun nativeRunLinearForward(
            handle: Long, paramWeight: Int, paramBias: Int,
            T: Int, inF: Int, outF: Int,
            input: FloatArray, output: FloatArray)
        @JvmStatic private external fun nativeRunLayerNormForward(
            handle: Long, paramGamma: Int, paramBeta: Int,
            T: Int, C: Int, eps: Float,
            input: FloatArray, output: FloatArray)
        @JvmStatic private external fun nativeStep(handle: Long, tokenIds: IntArray, targets: IntArray): Float
    }
}
