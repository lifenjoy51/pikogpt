package mps

import org.junit.jupiter.api.Assumptions
import kotlin.test.Test

/**
 * Phase 1 — MPSGraph JNI bridge skeleton 동작 검증.
 *
 * 1. dylib 로드 + nativeInit 성공
 * 2. createSession이 non-zero handle 반환
 * 3. destroySession 안전 (double-free 없음)
 *
 * 모델 graph build/run은 Phase 2+에서 별도 테스트.
 */
class MpsGraphSessionTest {

    private fun ensure() {
        val ok = MpsGraphSession.available()
        Assumptions.assumeTrue(ok, "MpsGraph unavailable: ${MpsGraphSession.loadError}")
    }

    @Test
    fun availabilityCheckPasses() {
        ensure()
    }

    @Test
    fun createAndDestroySession() {
        ensure()
        val config = MpsGraphConfig(
            numLayers = 16,
            embedDim = 256,
            numHeads = 8,
            blockSize = 32,
            vocab = 2000,
            batchSize = 2,
        )
        MpsGraphSession.create(config).use { /* opened then closed */ }
    }

    @Test
    fun loadWeightsRoundtrip() {
        ensure()
        val config = MpsGraphConfig(
            numLayers = 2,
            embedDim = 16,
            numHeads = 4,
            blockSize = 8,
            vocab = 64,
            batchSize = 1,
        )
        MpsGraphSession.create(config).use { s ->
            // 임의 weight 2개 등록. paramIndex 0=embedding, paramIndex 1=LN gamma.
            val emb = FloatArray(64 * 16) { it.toFloat() }
            s.loadWeights(0, emb, intArrayOf(64, 16))
            val gamma = FloatArray(16) { 1.0f }
            s.loadWeights(1, gamma, intArrayOf(16))
            check(s.weightCount() == 2) { "expected 2 weights, got ${s.weightCount()}" }
        }
    }

    @Test
    fun embeddingForwardMatchesNaive() {
        ensure()
        val vocab = 32
        val embedDim = 16
        val tokLen = 8
        val config = MpsGraphConfig(
            numLayers = 1, embedDim = embedDim, numHeads = 4,
            blockSize = tokLen, vocab = vocab, batchSize = 1,
        )
        // 임의 weight + tokens
        val embData = FloatArray(vocab * embedDim) { (it % 7 - 3).toFloat() / 7f }
        val tokens = IntArray(tokLen) { (it * 13 + 5) % vocab }

        MpsGraphSession.create(config).use { s ->
            s.loadWeights(0, embData, intArrayOf(vocab, embedDim))
            val out = FloatArray(tokLen * embedDim)
            s.runEmbeddingForward(tokens, out)
            // 검증: gather(emb, tok)와 일치
            for (i in 0 until tokLen) {
                val tok = tokens[i]
                for (j in 0 until embedDim) {
                    val expected = embData[tok * embedDim + j]
                    val actual = out[i * embedDim + j]
                    check(kotlin.math.abs(expected - actual) < 1e-5f) {
                        "mismatch at i=$i j=$j: expected=$expected actual=$actual"
                    }
                }
            }
        }
    }

    @Test
    fun layerNormForwardMatchesTurbo() {
        ensure()
        val T = 8
        val C = 16
        val config = MpsGraphConfig(
            numLayers = 1, embedDim = C, numHeads = 4,
            blockSize = T, vocab = 32, batchSize = 1,
        )
        val gamma = FloatArray(C) { 0.5f + (it % 5) * 0.1f }
        val beta = FloatArray(C) { -0.2f + (it % 3) * 0.05f }
        val input = FloatArray(T * C) { ((it * 7 + 3) % 17 - 8).toFloat() / 8f }

        // Turbo reference
        val turboLn = turbo.layer.TurboLayerNorm(C, useBias = true)
        System.arraycopy(gamma, 0, turboLn.gamma.data, 0, C)
        System.arraycopy(beta, 0, turboLn.beta.data, 0, C)
        val xT = turbo.TurboTensor(intArrayOf(T, C), input.copyOf())
        val yT = turboLn.forward(xT)

        // Mps result
        MpsGraphSession.create(config).use { s ->
            s.loadWeights(0, gamma, intArrayOf(C))
            s.loadWeights(1, beta, intArrayOf(C))
            val mpsOut = FloatArray(T * C)
            s.runLayerNormForward(0, 1, T, C, turboLn.eps, input, mpsOut)
            for (i in 0 until T * C) {
                check(kotlin.math.abs(yT.data[i] - mpsOut[i]) < 1e-4f) {
                    "LN mismatch at $i: turbo=${yT.data[i]} mps=${mpsOut[i]}"
                }
            }
        }
    }

    @Test
    fun linearForwardMatchesTurbo() {
        ensure()
        val T = 8
        val inF = 16
        val outF = 12
        val config = MpsGraphConfig(
            numLayers = 1, embedDim = inF, numHeads = 4,
            blockSize = T, vocab = 32, batchSize = 1,
        )
        val w = FloatArray(outF * inF) { ((it * 11 + 5) % 19 - 9).toFloat() / 19f }
        val b = FloatArray(outF) { (it - outF / 2).toFloat() * 0.05f }
        val input = FloatArray(T * inF) { ((it * 7 + 3) % 17 - 8).toFloat() / 17f }

        val turboLn = turbo.layer.TurboLinear(inF, outF, useBias = true)
        System.arraycopy(w, 0, turboLn.weight.data, 0, w.size)
        System.arraycopy(b, 0, turboLn.bias!!.data, 0, b.size)
        val yT = turboLn.forward(turbo.TurboTensor(intArrayOf(T, inF), input.copyOf()))

        MpsGraphSession.create(config).use { s ->
            s.loadWeights(0, w, intArrayOf(outF, inF))
            s.loadWeights(1, b, intArrayOf(outF))
            val mpsOut = FloatArray(T * outF)
            s.runLinearForward(0, 1, T, inF, outF, input, mpsOut)
            for (i in 0 until T * outF) {
                check(kotlin.math.abs(yT.data[i] - mpsOut[i]) < 1e-4f) {
                    "Linear mismatch at $i: turbo=${yT.data[i]} mps=${mpsOut[i]}"
                }
            }
        }
    }

    @Test
    fun swiGluForwardMatchesTurbo() {
        ensure()
        val T = 8
        val embedDim = 16
        // turbo SwiGLU hiddenDim 계산: 4 * embedDim → 8/3로 줄임. 확인 필요.
        // 일단 turbo TurboMLP를 인스턴스화해서 hiddenDim 가져옴.
        val turboMlp = turbo.layer.TurboMLP(embedDim, useBias = true, dropoutProbability = 0.0f, activation = "swiglu")
        val gateW = turboMlp.gateProjection!!.weight
        val gateB = turboMlp.gateProjection!!.bias!!
        val upW = turboMlp.upProjection!!.weight
        val upB = turboMlp.upProjection!!.bias!!
        val downW = turboMlp.downProjection!!.weight
        val downB = turboMlp.downProjection!!.bias!!
        val hiddenDim = gateW.rows  // [hiddenDim, embedDim]

        // 결정적 초기화로 weight 재기록
        val rng = kotlin.random.Random(42L)
        for (i in 0 until gateW.numel) gateW.data[i] = (rng.nextFloat() - 0.5f) * 0.5f
        for (i in 0 until gateB.numel) gateB.data[i] = (rng.nextFloat() - 0.5f) * 0.1f
        for (i in 0 until upW.numel) upW.data[i] = (rng.nextFloat() - 0.5f) * 0.5f
        for (i in 0 until upB.numel) upB.data[i] = (rng.nextFloat() - 0.5f) * 0.1f
        for (i in 0 until downW.numel) downW.data[i] = (rng.nextFloat() - 0.5f) * 0.5f
        for (i in 0 until downB.numel) downB.data[i] = (rng.nextFloat() - 0.5f) * 0.1f

        val inputData = FloatArray(T * embedDim) { (rng.nextFloat() - 0.5f) * 1.0f }

        val yT = turboMlp.forward(turbo.TurboTensor(intArrayOf(T, embedDim), inputData.copyOf()))

        val config = MpsGraphConfig(
            numLayers = 1, embedDim = embedDim, numHeads = 4,
            blockSize = T, vocab = 32, batchSize = 1,
        )
        MpsGraphSession.create(config).use { s ->
            s.loadWeights(0, gateW.data, intArrayOf(hiddenDim, embedDim))
            s.loadWeights(1, gateB.data, intArrayOf(hiddenDim))
            s.loadWeights(2, upW.data, intArrayOf(hiddenDim, embedDim))
            s.loadWeights(3, upB.data, intArrayOf(hiddenDim))
            s.loadWeights(4, downW.data, intArrayOf(embedDim, hiddenDim))
            s.loadWeights(5, downB.data, intArrayOf(embedDim))
            val mpsOut = FloatArray(T * embedDim)
            s.runSwiGluForward(0, 1, 2, 3, 4, 5, T, embedDim, hiddenDim, inputData, mpsOut)
            var maxDelta = 0f
            for (i in 0 until T * embedDim) {
                val d = kotlin.math.abs(yT.data[i] - mpsOut[i])
                if (d > maxDelta) maxDelta = d
            }
            check(maxDelta < 1e-3f) { "SwiGLU max delta $maxDelta exceeds 1e-3" }
        }
    }

    @Test
    fun attentionForwardMatchesTurbo() {
        ensure()
        val T = 6
        val embedDim = 16
        val numHeads = 4
        val headDim = embedDim / numHeads
        val half = headDim / 2

        val attn = turbo.layer.TurboSelfAttention(
            embedDim = embedDim,
            numHeads = numHeads,
            useBias = true,
            dropoutProbability = 0.0f,
            positionEncoding = "rope",
        )

        val rng = kotlin.random.Random(123L)
        fun fill(arr: FloatArray, scale: Float) {
            for (i in arr.indices) arr[i] = (rng.nextFloat() - 0.5f) * scale
        }
        fill(attn.qProjection!!.weight.data, 0.4f)
        fill(attn.qProjection!!.bias!!.data, 0.05f)
        fill(attn.kProjection!!.weight.data, 0.4f)
        fill(attn.kProjection!!.bias!!.data, 0.05f)
        fill(attn.vProjection!!.weight.data, 0.4f)
        fill(attn.vProjection!!.bias!!.data, 0.05f)
        fill(attn.outputProjection.weight.data, 0.4f)
        fill(attn.outputProjection.bias!!.data, 0.05f)

        val input = FloatArray(T * embedDim).also { fill(it, 1.0f) }
        val yT = attn.forward(turbo.TurboTensor(intArrayOf(T, embedDim), input.copyOf()))

        // cos/sin tables
        val cos = FloatArray(T * half)
        val sin = FloatArray(T * half)
        for (t in 0 until T) {
            for (i in 0 until half) {
                val theta = Math.pow(10000.0, -2.0 * i / headDim)
                val angle = t * theta
                cos[t * half + i] = kotlin.math.cos(angle).toFloat()
                sin[t * half + i] = kotlin.math.sin(angle).toFloat()
            }
        }
        // mask: lower-tri 0, upper -1e9
        val mask = FloatArray(T * T)
        for (i in 0 until T) for (j in 0 until T) mask[i * T + j] = if (j > i) -1e9f else 0f

        val config = MpsGraphConfig(
            numLayers = 1, embedDim = embedDim, numHeads = numHeads,
            blockSize = T, vocab = 32, batchSize = 1,
        )
        MpsGraphSession.create(config).use { s ->
            s.loadWeights(0, attn.qProjection!!.weight.data, intArrayOf(embedDim, embedDim))
            s.loadWeights(1, attn.qProjection!!.bias!!.data, intArrayOf(embedDim))
            s.loadWeights(2, attn.kProjection!!.weight.data, intArrayOf(embedDim, embedDim))
            s.loadWeights(3, attn.kProjection!!.bias!!.data, intArrayOf(embedDim))
            s.loadWeights(4, attn.vProjection!!.weight.data, intArrayOf(embedDim, embedDim))
            s.loadWeights(5, attn.vProjection!!.bias!!.data, intArrayOf(embedDim))
            s.loadWeights(6, attn.outputProjection.weight.data, intArrayOf(embedDim, embedDim))
            s.loadWeights(7, attn.outputProjection.bias!!.data, intArrayOf(embedDim))

            val mpsOut = FloatArray(T * embedDim)
            s.runAttentionForward(0, 1, 2, 3, 4, 5, 6, 7,
                T, embedDim, numHeads, input, cos, sin, mask, mpsOut)

            var maxDelta = 0f
            for (i in 0 until T * embedDim) {
                val d = kotlin.math.abs(yT.data[i] - mpsOut[i])
                if (d > maxDelta) maxDelta = d
            }
            check(maxDelta < 2e-3f) { "Attention max delta $maxDelta exceeds 2e-3" }
        }
    }

    @Test
    fun fullForwardMatchesTurbo() {
        ensure()
        val T = 6
        val embedDim = 16
        val numHeads = 4
        val numLayers = 2
        val vocab = 32
        val headDim = embedDim / numHeads
        val half = headDim / 2

        // turbo 모델 생성 (RoPE + SwiGLU + tied + LayerNorm)
        val gptCfg = gpt.GPTConfig(
            vocabularySize = vocab,
            embeddingDimension = embedDim,
            numberOfLayers = numLayers,
            numberOfAttentionHeads = numHeads,
            maxSequenceLength = T,
            dropoutProbability = 0.0f,
            useBias = true,
        )
        val modelCfg = turbo.TurboModelConfig(
            gpt = gptCfg,
            tieWeights = true,
            mlpActivation = "swiglu",
            positionEncoding = "rope",
            normalizationType = "layernorm",
        )
        val tm = turbo.layer.TurboPikoGPT(modelCfg)

        // 결정적 초기화
        val rng = kotlin.random.Random(7L)
        for (p in tm.parameters()) {
            for (i in 0 until p.numel) p.data[i] = (rng.nextFloat() - 0.5f) * 0.2f
        }
        // tokenIds
        val tokens = IntArray(T) { (rng.nextInt(vocab)) }

        val yT = tm.forward(tokens)  // [T, vocab]

        // cos/sin/mask
        val cos = FloatArray(T * half)
        val sin = FloatArray(T * half)
        for (t in 0 until T) for (i in 0 until half) {
            val theta = Math.pow(10000.0, -2.0 * i / headDim)
            val angle = t * theta
            cos[t * half + i] = kotlin.math.cos(angle).toFloat()
            sin[t * half + i] = kotlin.math.sin(angle).toFloat()
        }
        val mask = FloatArray(T * T)
        for (i in 0 until T) for (j in 0 until T) mask[i * T + j] = if (j > i) -1e9f else 0f

        val config = MpsGraphConfig(
            numLayers = numLayers, embedDim = embedDim, numHeads = numHeads,
            blockSize = T, vocab = vocab, batchSize = 1,
        )
        MpsGraphSession.create(config).use { s ->
            // 모든 parameter를 paramIndex 순서로 loadWeights
            val params = tm.parameters()
            for ((idx, p) in params.withIndex()) {
                s.loadWeights(idx, p.data, p.shape)
            }
            check(s.weightCount() == params.size) {
                "weights ${s.weightCount()} != turbo params ${params.size}"
            }
            val logits = FloatArray(T * vocab)
            s.runFullForward(tokens, cos, sin, mask, logits)
            var maxDelta = 0f
            for (i in 0 until T * vocab) {
                val d = kotlin.math.abs(yT.data[i] - logits[i])
                if (d > maxDelta) maxDelta = d
            }
            check(maxDelta < 5e-3f) { "Full forward max delta $maxDelta exceeds 5e-3" }
        }
    }

    @Test
    fun trainingStepReducesLoss() {
        ensure()
        val T = 8
        val embedDim = 16
        val numHeads = 4
        val numLayers = 1
        val vocab = 32
        val headDim = embedDim / numHeads
        val half = headDim / 2

        val gptCfg = gpt.GPTConfig(
            vocabularySize = vocab,
            embeddingDimension = embedDim,
            numberOfLayers = numLayers,
            numberOfAttentionHeads = numHeads,
            maxSequenceLength = T,
            dropoutProbability = 0.0f,
            useBias = true,
        )
        val modelCfg = turbo.TurboModelConfig(
            gpt = gptCfg, tieWeights = true, mlpActivation = "swiglu",
            positionEncoding = "rope", normalizationType = "layernorm",
        )
        val tm = turbo.layer.TurboPikoGPT(modelCfg)
        val rng = kotlin.random.Random(11L)
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

        val config = MpsGraphConfig(numLayers, embedDim, numHeads, T, vocab, batchSize = 1)
        MpsGraphSession.create(config).use { s ->
            for ((idx, p) in tm.parameters().withIndex()) {
                s.loadWeights(idx, p.data, p.shape)
            }
            var firstLoss = 0f
            var lastLoss = 0f
            for (step in 1..20) {
                val l = s.runTrainingStep(tokens, targets, cos, sin, mask,
                    lr = 1e-2f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f,
                    weightDecay = 0.0f, stepT = step)
                if (step == 1) firstLoss = l
                lastLoss = l
            }
            check(lastLoss < firstLoss - 0.1f) {
                "loss 감소 안 함: first=$firstLoss last=$lastLoss"
            }
        }
    }

    @Test
    fun doubleCloseIsSafe() {
        ensure()
        val config = MpsGraphConfig(
            numLayers = 4,
            embedDim = 64,
            numHeads = 4,
            blockSize = 16,
            vocab = 256,
            batchSize = 1,
        )
        val s = MpsGraphSession.create(config)
        s.close()
        s.close()
    }
}
