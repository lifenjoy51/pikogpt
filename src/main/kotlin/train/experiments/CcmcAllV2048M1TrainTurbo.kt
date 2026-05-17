package train.experiments

import turbo.TurboTrainConfig
import turbo.TurboTrainer

/**
 * ccmc-all-v2048-v8 (vocab=2048, 7파일 corpus, BOS/EOS, heads=3) ~986k tied 모델 turbo — 160000 iter.
 *
 * v7(485k, heads=2) 대비 변경:
 *   - embeddingDimension 64 → 96 (1.5×, 폭 확장)
 *   - numberOfHeads 2 → 3 (head_dim 32 유지)
 *   - learningRate 3e-4 → 2e-4 (큰 모델 stability)
 *   - dropout 0.1 → 0.05 (capacity 늘었으므로 regularization 완화)
 *   - gradientAccumulationSteps 4 → 8 (effective batch 32 → 64, plateau fluctuation 완화)
 *   - beta2 0.95 → 0.99 (Adam 2차 모멘트 smoothing 강화)
 *   - maxIters 300k → 160k (iter rate ~0.27s × 160k = 12h 예산 맞춤)
 *
 * 모델: embd=96, layers=7, heads=3, tied=true → ~985,824 params
 *   - Token emb (2048×96): 196,608 (20.0%)
 *   - Pos emb (64×96): 6,144
 *   - Transformer × 7 layers: 782,880 (79.4%) — block당 111,840
 *     · attention: 4d²+4d = 37,248
 *     · 2 LN:      4d = 384
 *     · MLP GELU:  8d²+5d = 74,208
 *   - Final LN: 192
 *   - LM head: 0 (tied)
 *
 * Data: v7과 동일 (data/ccmc-all-v2048-v8/, train ~6.97M tokens / val ~366k tokens)
 *       BOS/EOS 래핑, lemma_sentences 문장 분리, vocab=2048.
 *
 * 평가: 매 1600 iter (evalIntervalRatio=0.01 → 최대 100 ckpt)
 * Early stop: patience=20.
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TurboTrainConfig(
        dataPath = "data/ccmc-all-v2048-v8",
        modelDir = "model",
        expName = "main",
        gradientAccumulationSteps = 8,
        batchSize = 8,
        blockSize = 64,
        numberOfLayers = 7,
        numberOfHeads = 3,
        embeddingDimension = 96,
        dropout = 0.05f,
        bias = true,
        learningRate = 2e-4f,
        weightDecay = 0.01f,
        gradClip = 1.0f,
        beta1 = 0.9f,
        beta2 = 0.99f,
        maxIters = maxItersOverride ?: 160000,
        warmupRatio = 0.05f,
        learningRateDecayRatio = 0.95f,
        minimumLearningRate = 1e-5f,
        decayLr = true,
        evalIntervalRatio = 0.01f,
        evalIters = 10,
        logInterval = 500,
        alwaysSaveCheckpoint = true,
        earlyStopPatience = 20,
        initFrom = if (resume) "resume" else "scratch",
        modelCheckpointDir = null,
        tieWeights = true,
    )

    TurboTrainer(config).train()
}
