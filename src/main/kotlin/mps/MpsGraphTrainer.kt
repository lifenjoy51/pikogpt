package mps

import train.DataLoader
import turbo.TurboModelConfig
import turbo.layer.TurboPikoGPT
import java.io.File

/**
 * P0.5 — MPSGraph backend 학습 진입점.
 *
 * TurboTrainer 수준 기능:
 *   - LR scheduler (warmup + cosine decay) — P0.2
 *   - eval 매 evalInterval + train/val loss 평균 — P0.3
 *   - best loss 추적 + 갱신 시 체크포인트 — P0.1, P0.3
 *   - gradient clipping (graph 내) — P0.4
 *   - resume from latest checkpoint
 *   - early stop (patience / plateau) — P0.3
 *
 * PoC 단계 한계: batchSize=1 (P1.1에서 일반화), gradient accumulation 없음 (P1.2).
 */
class MpsGraphTrainer(
    val trainConfig: MpsGraphTrainConfig,
    val turboModelConfig: TurboModelConfig,
) {
    private val datasetName: String = File(trainConfig.dataPath).name
    private val expPath: File = File(File(trainConfig.modelDir, datasetName), trainConfig.expName)

    private val graphConfig = MpsGraphConfig(
        numLayers = turboModelConfig.gpt.numberOfLayers,
        embedDim = turboModelConfig.gpt.embeddingDimension,
        numHeads = turboModelConfig.gpt.numberOfAttentionHeads,
        blockSize = trainConfig.blockSize,
        vocab = turboModelConfig.gpt.vocabularySize,
        batchSize = trainConfig.batchSize,
        useSwiglu = turboModelConfig.mlpActivation == "swiglu",
        useRope = turboModelConfig.positionEncoding == "rope",
        tieWeights = turboModelConfig.tieWeights,
        useFp16 = trainConfig.useFp16,
        useDropout = trainConfig.dropoutProbability > 0.0f,
        dropoutProbability = trainConfig.dropoutProbability,
        gradientAccumulationSteps = trainConfig.gradientAccumulationSteps,
        // useVariableForStep는 trainer가 accum/adam 분리 path를 사용하므로 false 유지.
    )

    fun train() {
        require(MpsGraphSession.available()) {
            "MpsGraph unavailable: ${MpsGraphSession.loadError}"
        }
        if (trainConfig.useFp16) {
            println("[mps-graph] useFp16=true → stepGraph forward fp16, LN은 fp32 cast 유지 (안정성).")
        }
        if (trainConfig.useExecutableCache && trainConfig.executableCachePath != null) {
            println("[mps-graph] useExecutableCache=true path=${trainConfig.executableCachePath}: compile + disk roundtrip은 API로 노출됨. run path는 graph 기반 유지 (PoC).")
        }

        val T = trainConfig.blockSize
        val headDim = graphConfig.embedDim / graphConfig.numHeads
        val half = headDim / 2

        val cos = FloatArray(T * half)
        val sin = FloatArray(T * half)
        for (t in 0 until T) for (i in 0 until half) {
            val theta = Math.pow(10000.0, -2.0 * i / headDim)
            cos[t * half + i] = kotlin.math.cos(t * theta).toFloat()
            sin[t * half + i] = kotlin.math.sin(t * theta).toFloat()
        }
        val mask = FloatArray(T * T) { idx -> if (idx % T > idx / T) -1e9f else 0f }

        val trainBin = File(trainConfig.dataPath, "train.bin")
        val valBinFile = File(trainConfig.dataPath, "val.bin")
        val B = trainConfig.batchSize
        val trainLoader = DataLoader(trainBin.absolutePath, batchSize = B, blockSize = T)
        val valLoader = if (valBinFile.exists())
            DataLoader(valBinFile.absolutePath, batchSize = B, blockSize = T) else trainLoader
        // Phase 1 — train batch는 별도 thread가 한 step 앞서 prepare (GPU 실행 중 host가 다음 batch 준비).
        val trainPrefetcher = BatchPrefetcher(trainLoader, T)

        // turbo 모델로 weight init (random or zeros). scratch에서만 사용.
        val tm = TurboPikoGPT(turboModelConfig)
        val params = tm.parameters()
        val shapes: List<IntArray> = params.map { it.shape }
        println("[mps-graph] 모델 파라미터: tensor=${params.size}, 총 스칼라=${params.sumOf { it.numel }}")

        val session = MpsGraphSession.create(graphConfig)
        var startIter = 0
        var initialBestLoss = Double.POSITIVE_INFINITY

        try {
            // 가중치 초기화
            for ((idx, p) in params.withIndex()) {
                session.loadWeights(idx, p.data, p.shape)
            }
            // P1.2 — grad accumulator는 0에서 시작
            session.resetGradAccum()

            // resume
            if (trainConfig.initFrom == "resume") {
                val latest = findLatestCheckpoint()
                if (latest != null) {
                    val meta = MpsCheckpointIO.load(session, shapes, latest)
                    startIter = meta.iterationNumber
                    initialBestLoss = meta.bestValidationLoss
                    println("[mps-graph] resume: ${latest.absolutePath} (iter=$startIter, bestLoss=${"%.4f".format(initialBestLoss)})")
                } else {
                    println("[mps-graph] resume 요청했지만 ckpt 없음 — scratch로 시작")
                }
            }

            val tracker = MpsBestLossTracker(
                smoothingWindow = 5,
                earlyStopPatience = trainConfig.earlyStopPatience,
                initialBestLoss = initialBestLoss,
            )

            val startTime = System.currentTimeMillis()
            var runningSum = 0.0
            var runningCount = 0
            val evalInterval = trainConfig.evalInterval
            val warmupIters = trainConfig.warmupIters
            val decayIters = trainConfig.learningRateDecayIterations

            var iter = startIter
            outer@ while (iter < trainConfig.maxIters) {
                if (iter > 0 && iter % evalInterval == 0) {
                    val trainAvg = evalLoss(session, trainLoader, cos, sin, mask, trainConfig.evalIters)
                    val valAvg = evalLoss(session, valLoader, cos, sin, mask, trainConfig.evalIters)
                    val combined = (trainAvg + valAvg) / 2.0
                    val r = tracker.update(combined)
                    println(
                        "스텝 $iter: 훈련 ${"%.4f".format(trainAvg)} | " +
                            "검증 ${"%.4f".format(valAvg)} | 평균 ${"%.4f".format(combined)} | " +
                            "smoothed ${"%.4f".format(r.smoothedLoss)}" +
                            (if (r.isBest) " | BEST" else "")
                    )

                    if (r.isBest || trainConfig.alwaysSaveCheckpoint) {
                        saveCheckpoint(session, shapes, iter, tracker.bestLoss, r.isBest)
                    }
                    if (r.shouldStopByPatience) {
                        println("Early stop (patience): best 갱신 ${trainConfig.earlyStopPatience}회 연속 없음 — iter=$iter")
                        break@outer
                    }
                    if (r.shouldStopByPlateau) {
                        println("Early stop (plateau): iter=$iter")
                        break@outer
                    }
                }

                val lr = MpsLrSchedule.computeLr(
                    iter = iter,
                    warmupIters = warmupIters,
                    decayIters = decayIters,
                    baseLr = trainConfig.learningRate,
                    minLr = trainConfig.minimumLearningRate,
                    decayEnabled = trainConfig.decayLr,
                )
                // P1.2 진정한 grad accumulation — accumGraph N번 + adamGraph 1번 = 1 effective iter.
                // gradAccum=1이면 1 accum + 1 adam (수학적으로 step graph와 동등).
                // Phase 3 — fused step: 8 accum + 1 adam을 단일 commandBuffer commit.
                val accumSteps = trainConfig.gradientAccumulationSteps.coerceAtLeast(1)
                val allTokens = IntArray(accumSteps * B * T)
                val allTargets = IntArray(accumSteps * B * T)
                val L = graphConfig.numLayers
                val E = graphConfig.embedDim
                val maskNumel = 2 * L * B * T * E
                val allDpMask: FloatArray? =
                    if (trainConfig.dropoutProbability > 0f) FloatArray(accumSteps * maskNumel) else null
                for (micro in 0 until accumSteps) {
                    val (flatInputs, flatTargets) = trainPrefetcher.next()
                    System.arraycopy(flatInputs, 0, allTokens, micro * B * T, B * T)
                    System.arraycopy(flatTargets, 0, allTargets, micro * B * T, B * T)
                    if (allDpMask != null) {
                        val dp = makeDropoutMaskOrNull(B, T)!!
                        System.arraycopy(dp, 0, allDpMask, micro * maskNumel, maskNumel)
                    }
                }
                val loss = session.runFusedStep(
                    allTokens, allTargets, cos, sin, mask, allDpMask,
                    lr = lr, beta1 = trainConfig.beta1, beta2 = trainConfig.beta2,
                    eps = trainConfig.eps, weightDecay = trainConfig.weightDecay,
                    gradClip = trainConfig.gradClip, stepT = iter + 1,
                    batchSize = B, accumSteps = accumSteps,
                )
                runningSum += loss
                runningCount += 1

                if (iter % trainConfig.logInterval == 0) {
                    val running = if (runningCount > 0) runningSum / runningCount else loss.toDouble()
                    val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
                    println("iter $iter: loss=${"%.4f".format(loss)} running=${"%.4f".format(running)} lr=${"%.2e".format(lr)} elapsed=${"%.1f".format(elapsed)}s")
                    runningSum = 0.0
                    runningCount = 0
                }
                iter += 1
            }
            println("[mps-graph] 학습 종료. 최종 iter=$iter, bestLoss=${"%.4f".format(tracker.bestLoss)}")
        } finally {
            trainPrefetcher.close()
            session.close()
        }
    }

    private fun evalLoss(
        session: MpsGraphSession,
        loader: DataLoader,
        cos: FloatArray, sin: FloatArray, mask: FloatArray,
        iters: Int,
    ): Double {
        // P1.1 — batch 단위 1회 호출로 평가. loss는 B*T mean (각 sample 가중 동일).
        val B = trainConfig.batchSize
        val T = trainConfig.blockSize
        var sum = 0.0
        var count = 0
        repeat(iters) {
            val (inp, tgt) = loader.getBatch()
            val flatInp = flattenBatch(inp, T)
            val flatTgt = flattenBatch(tgt, T)
            val l = session.runForwardLoss(flatInp, flatTgt, cos, sin, mask, batchSize = B)
            sum += l.toDouble()
            count += 1
        }
        return if (count > 0) sum / count else Double.NaN
    }

    private fun flattenBatch(rows: Array<IntArray>, T: Int): IntArray {
        val B = rows.size
        val out = IntArray(B * T)
        for (b in 0 until B) {
            require(rows[b].size == T) { "row[$b] size ${rows[b].size} != T $T" }
            System.arraycopy(rows[b], 0, out, b * T, T)
        }
        return out
    }

    // P4 — training-time dropout mask [2*numLayers, B, T, embedDim].
    //   inverted dropout: keep with prob (1-p) → value 1/(1-p), drop → 0.
    //   probability ≤ 0이면 null 반환 (mps backend가 dropout placeholder 자체를 안 만들거나 mask=1로 처리).
    // Phase 1 — IntStream.parallel + ThreadLocalRandom로 host CPU 8 코어 활용 (single-thread bottleneck 해소).
    private fun makeDropoutMaskOrNull(B: Int, T: Int): FloatArray? {
        val p = trainConfig.dropoutProbability
        if (p <= 0.0f) return null
        val L = graphConfig.numLayers
        val E = graphConfig.embedDim
        val size = 2 * L * B * T * E
        val keep = 1.0f / (1.0f - p)
        val out = FloatArray(size)
        java.util.stream.IntStream.range(0, size).parallel().forEach { i ->
            out[i] = if (java.util.concurrent.ThreadLocalRandom.current().nextFloat() < p) 0.0f else keep
        }
        return out
    }

    // Phase 1 — train batch prefetcher. GPU 실행 중 host가 다음 batch를 별도 thread에서 준비.
    // queue 용량 2 = 1 ready + 1 in-flight; main thread는 next()로 ready batch 즉시 받음.
    private class BatchPrefetcher(
        private val loader: DataLoader,
        private val T: Int,
    ) : AutoCloseable {
        private val queue = java.util.concurrent.LinkedBlockingQueue<Pair<IntArray, IntArray>>(2)
        @Volatile private var stopFlag = false
        private val thread = Thread({
            try {
                while (!stopFlag) {
                    val (inp, tgt) = loader.getBatch()
                    val B = inp.size
                    val flatInp = IntArray(B * T)
                    val flatTgt = IntArray(B * T)
                    for (b in 0 until B) {
                        System.arraycopy(inp[b], 0, flatInp, b * T, T)
                        System.arraycopy(tgt[b], 0, flatTgt, b * T, T)
                    }
                    queue.put(flatInp to flatTgt)
                }
            } catch (_: InterruptedException) { /* shutdown */ }
        }, "mps-graph-batch-prefetcher").apply { isDaemon = true; start() }

        fun next(): Pair<IntArray, IntArray> = queue.take()
        override fun close() { stopFlag = true; thread.interrupt() }
    }

    private fun saveCheckpoint(
        session: MpsGraphSession,
        shapes: List<IntArray>,
        iter: Int,
        bestLoss: Double,
        isBest: Boolean,
    ) {
        val nextVersion = nextCheckpointVersion()
        val dir = File(expPath, "v%04d".format(nextVersion))
        val meta = MpsCheckpoint(
            iterationNumber = iter,
            bestValidationLoss = bestLoss,
            modelArgs = turboModelConfig,
        )
        MpsCheckpointIO.save(session, shapes, dir, meta)
        val srcMeta = File(trainConfig.dataPath, "meta.json")
        if (srcMeta.exists()) {
            srcMeta.copyTo(File(dir, "meta.json"), overwrite = true)
        }
        val label = if (isBest) "best" else "always"
        println("체크포인트 저장 ($label): ${File(dir, "checkpoint.json").absolutePath}")
    }

    private fun nextCheckpointVersion(): Int {
        if (!expPath.exists()) return 1
        val rx = Regex("^v(\\d+)$")
        val maxV = (expPath.listFiles { f -> f.isDirectory } ?: emptyArray())
            .mapNotNull { rx.matchEntire(it.name)?.groupValues?.get(1)?.toIntOrNull() }
            .maxOrNull() ?: 0
        return maxV + 1
    }

    private fun findLatestCheckpoint(): File? {
        if (!expPath.exists()) return null
        val rx = Regex("^v(\\d+)$")
        val dirs = expPath.listFiles { f -> f.isDirectory } ?: emptyArray()
        var best: File? = null
        var bestV = -1
        for (d in dirs) {
            val m = rx.matchEntire(d.name) ?: continue
            val v = m.groupValues[1].toIntOrNull() ?: continue
            if (File(d, "checkpoint.json").exists() && File(d, "model_weights.bin").exists() && v > bestV) {
                bestV = v
                best = d
            }
        }
        return best
    }
}
