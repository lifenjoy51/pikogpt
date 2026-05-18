package train.experiments

import mps.MpsGraphConfig
import mps.MpsGraphSession
import train.DataLoader
import kotlin.system.measureTimeMillis

/**
 * Phase 5 — MPSGraph 기반 GPU 100% training 진입점.
 *
 * Bench10MTurbo와 같은 모델 hyperparam (16 layer, embedDim=256, 8 heads, blockSize=32, vocab=2000).
 * 차이점:
 *   - batchSize=1, gradAccum=1 (PoC, GPU 1 dispatch per step)
 *   - TurboParallel 사용 안 함 (GPU 자체가 throughput)
 *   - forward + backward + AdamW를 단일 GPU graph로 실행
 *
 * 매 iter wall-clock 측정 + 마지막 iter 누적 시간 출력.
 */
fun main(args: Array<String>) {
    if (!MpsGraphSession.available()) {
        println("[mps-graph] unavailable: ${MpsGraphSession.loadError}")
        return
    }
    val maxIters = args.getOrNull(0)?.toIntOrNull() ?: 50
    // P1.1 — args[1]: batchSize. 기본 1.
    val batchSize = args.getOrNull(1)?.toIntOrNull() ?: 1

    val embedDim = 256
    val numLayers = 16
    val numHeads = 8
    val blockSize = 32
    val vocab = 2000
    val headDim = embedDim / numHeads
    val half = headDim / 2

    val gptCfg = gpt.GPTConfig(
        vocabularySize = vocab, embeddingDimension = embedDim,
        numberOfLayers = numLayers, numberOfAttentionHeads = numHeads,
        maxSequenceLength = blockSize, dropoutProbability = 0.0f, useBias = true,
    )
    val modelCfg = turbo.TurboModelConfig(
        gpt = gptCfg, tieWeights = true, mlpActivation = "swiglu",
        positionEncoding = "rope", normalizationType = "layernorm",
    )
    // weight init용 turbo 모델 (랜덤 초기화). 학습은 mps graph가 함.
    val tm = turbo.layer.TurboPikoGPT(modelCfg)
    val params = tm.parameters()
    println("모델 파라미터 텐서 수: ${params.size}, 총 스칼라: ${params.sumOf { it.numel }}")

    val cosTable = FloatArray(blockSize * half)
    val sinTable = FloatArray(blockSize * half)
    for (t in 0 until blockSize) for (i in 0 until half) {
        val theta = Math.pow(10000.0, -2.0 * i / headDim)
        cosTable[t * half + i] = kotlin.math.cos(t * theta).toFloat()
        sinTable[t * half + i] = kotlin.math.sin(t * theta).toFloat()
    }
    val mask = FloatArray(blockSize * blockSize) { idx ->
        if (idx % blockSize > idx / blockSize) -1e9f else 0f
    }

    val trainBin = "data/ccmc-v2-pro/stage2/train.bin"
    val loader = DataLoader(trainBin, batchSize, blockSize)

    val session = MpsGraphSession.create(
        MpsGraphConfig(numLayers, embedDim, numHeads, blockSize, vocab, batchSize)
    )
    for ((idx, p) in params.withIndex()) {
        session.loadWeights(idx, p.data, p.shape)
    }
    println("[mps-graph] weights 로드 완료 (${params.size}개 tensor, GPU resident)")

    val startTime = System.currentTimeMillis()
    val lr = 3e-4f
    val beta1 = 0.9f
    val beta2 = 0.95f
    val eps = 1e-8f
    val wd = 0.01f

    var lastLoss = 0f
    for (iter in 1..maxIters) {
        val (inputs, targets) = loader.getBatch()
        val flatIn = IntArray(batchSize * blockSize)
        val flatTg = IntArray(batchSize * blockSize)
        for (b in 0 until batchSize) {
            System.arraycopy(inputs[b], 0, flatIn, b * blockSize, blockSize)
            System.arraycopy(targets[b], 0, flatTg, b * blockSize, blockSize)
        }
        val stepMs = measureTimeMillis {
            lastLoss = session.runTrainingStep(
                flatIn, flatTg, cosTable, sinTable, mask,
                lr, beta1, beta2, eps, wd, iter,
                gradClip = 0f, batchSize = batchSize,
            )
        }
        if (iter == 1 || iter % 10 == 0 || iter == maxIters) {
            val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
            println("iter $iter: loss=${"%.4f".format(lastLoss)}, step_ms=$stepMs, total=${"%.1f".format(elapsed)}s")
        }
    }
    session.close()

    val totalMs = System.currentTimeMillis() - startTime
    val tokensPerIter = batchSize * blockSize
    val totalTokens = tokensPerIter * maxIters
    println("\n=== Bench10MMpsGraph 완료 (batchSize=$batchSize) ===")
    println("총 시간: ${totalMs}ms (${maxIters} iter, ${totalTokens} tokens)")
    println("iter당 평균: ${totalMs / maxIters}ms")
    println("tokens/sec: ${"%.1f".format(totalTokens * 1000.0 / totalMs)}")
}
