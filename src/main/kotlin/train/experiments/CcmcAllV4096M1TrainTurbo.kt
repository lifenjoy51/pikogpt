package train.experiments

import turbo.TurboTrainConfig
import turbo.TurboTrainer

/**
 * ccmc-all-v4096-v1 (vocab=4096, 7파일 corpus, BOS/EOS, heads=3) ~1.07M tied 모델 turbo — 120000 iter.
 *
 * v8(986k, vocab=2048) 대비 변경:
 *   - vocab 2048 → 4096 (token emb 2배 — rare entity 표현력 확장)
 *   - numberOfLayers 7 → 6 (token emb 확장 보상, ~1M params 유지)
 *   - maxIters 160k → 120k (~14h 예산: iter rate ~0.42s × 120k)
 *   - 나머지 hyperparam 전부 v8과 동일 (lr 2e-4, dropout 0.05, β₂ 0.99, gradAccum 8, batch 8, blockSize 64, heads 3)
 *
 * 모델: embd=96, layers=6, heads=3, vocab=4096, tied=true → ~1,070,592 params
 *   - Token emb (4096×96): 393,216 (36.7%)
 *   - Pos emb (64×96): 6,144
 *   - Transformer × 6 layers: 671,040 (62.7%) — block당 111,840
 *     · attention: 4d²+4d = 37,248
 *     · 2 LN:      4d = 384
 *     · MLP GELU:  8d²+5d = 74,208
 *   - Final LN: 192
 *   - LM head: 0 (tied)
 *
 * Data: data/ccmc-all-v4096-v1/ (train 6.26M tokens / val 328k tokens)
 *       BOS/EOS 래핑, lemma_sentences 문장 분리, vocab=4096.
 *
 * 평가: 매 1200 iter (evalIntervalRatio=0.01 → 최대 100 ckpt)
 * Early stop: patience=20.
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TurboTrainConfig(
        dataPath = "data/ccmc-all-v4096-v1",
        modelDir = "model",
        expName = "main",
        gradientAccumulationSteps = 8,
        batchSize = 8,
        blockSize = 64,
        numberOfLayers = 6,
        numberOfHeads = 3,
        embeddingDimension = 96,
        dropout = 0.05f,
        bias = true,
        learningRate = 2e-4f,
        weightDecay = 0.01f,
        gradClip = 1.0f,
        beta1 = 0.9f,
        beta2 = 0.99f,
        maxIters = maxItersOverride ?: 120000,
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
