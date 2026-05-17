package train.experiments

import turbo.TurboTrainConfig
import turbo.TurboTrainer

/**
 * ccmc-all-v2048-v6 (vocab=2048, 7파일 corpus, hapax 단순화) ~485k tied 모델 turbo — 300000 iter.
 *
 * 모델: embd=64, layers=7, heads=1, tied=true → ~485,184 params
 *   - Token emb (2048×64): 131,072 (27.0%)
 *   - Pos emb (64×64): 4,096
 *   - Transformer × 7 layers: 349,888 (72.1%) — block당 49,984
 *     · attention: 4d²+4d = 16,640
 *     · 2 LN:      4d = 256
 *     · MLP GELU:  8d²+5d = 33,088
 *   - Final LN: 128
 *   - LM head: 0 (tied)
 *
 * v2048(299k) 대비: embd 48 → 64 (1.33× 폭). depth/heads/blockSize 동일. params 1.62×.
 *
 * Data: ccmc-all-raw 7파일 (lemma+stories+dialogues+wiki+cause_seq+chained+counting) 합본
 *       95:5 split (seed=51), train ~6.40M tokens, val ~336k tokens.
 *       lemma_sentences는 문장 단위 분리 (3,209 lemma → 78,958 sentences).
 *
 * 평가: 매 3000 iter (evalIntervalRatio=0.01 → 최대 100 ckpt)
 * Early stop: patience=20 (best 갱신 없이 60k iter 시 종료).
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TurboTrainConfig(
        dataPath = "data/ccmc-all-v2048-v6",
        modelDir = "model",
        expName = "main",
        gradientAccumulationSteps = 4,
        batchSize = 8,
        blockSize = 64,
        numberOfLayers = 7,
        numberOfHeads = 1,
        embeddingDimension = 64,
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
