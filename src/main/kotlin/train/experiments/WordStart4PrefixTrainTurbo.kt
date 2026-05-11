package train.experiments

import turbo.TurboTrainConfig
import turbo.TurboTrainer

/**
 * ccmc-all-v2048 (vocab=2048) ~299k tied 모델 turbo — 300000 iter, early stop patience=20.
 *
 * 모델: embd=48, layers=7, heads=1, tied=true → 299,376 params
 *   - Token emb (2048×48): 98,304 (32.8%)
 *   - Pos emb (64×48): 3,072
 *   - Transformer × 7 layers: 197,904 (66.1%)
 *   - Final LN: 96
 *   - LM head: 0 (tied)
 * v1024(111k) 대비 transformer 코어 2.59배(76,224→197,904), embd 1.5배(32→48), L +1(6→7).
 * Data: ccmc-all-raw 4파일 합본(lemma+stories+dialogues+wiki) 95:5 split, train 5.03M tokens.
 * 평가: 매 3000 iter (evalIntervalRatio=0.01 → 최대 100 ckpt)
 * Early stop: best 갱신 없이 20회 연속 eval (=60k iter) 시 종료. v1024 plateau가 ~22 eval에서 발생한 점 반영.
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TurboTrainConfig(
        dataPath = "data/ccmc-all-v2048",
        modelDir = "model",
        expName = "main",
        gradientAccumulationSteps = 4,
        batchSize = 8,
        blockSize = 64,
        numberOfLayers = 7,
        numberOfHeads = 1,
        embeddingDimension = 48,
        dropout = 0.1f,
        bias = true,
        learningRate = 3e-4f,
        weightDecay = 0.01f,
        gradClip = 1.0f,
        beta1 = 0.9f,
        beta2 = 0.95f,
        maxIters = maxItersOverride ?: 300000,
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
