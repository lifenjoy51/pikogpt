package train.experiments

import turbo.TurboTrainConfig
import turbo.TurboTrainer

/**
 * ccmc-all-v2048-v7 (vocab=2048, 7파일 corpus, BOS/EOS 주입, heads=2) ~485k tied 모델 turbo — 300000 iter.
 *
 * v6 wider(heads=1) 대비 변경:
 *   1) corpus 각 record를 `<|bos|> ... <|eos|>`로 래핑 → 모델이 record 종결을 명시적 토큰으로 학습
 *   2) numberOfHeads 1 → 2 (head_dim 64 → 32)
 *
 * 모델: embd=64, layers=7, heads=2, tied=true → ~485,184 params (heads는 attention 분할 방식만 바꿈, 총 params 동일)
 *   - Token emb (2048×64): 131,072 (27.0%)
 *   - Pos emb (64×64): 4,096
 *   - Transformer × 7 layers: 349,888 (72.1%) — block당 49,984
 *   - Final LN: 128
 *   - LM head: 0 (tied)
 *
 * Data: ccmc-all-raw 7파일 + lemma_sentences 문장 분리 + BOS/EOS 래핑
 *       95:5 split (seed=51), train ~6.97M tokens, val ~366k tokens.
 *
 * 평가: 매 3000 iter (evalIntervalRatio=0.01 → 최대 100 ckpt)
 * Early stop: patience=20.
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TurboTrainConfig(
        dataPath = "data/ccmc-all-v2048-v7",
        modelDir = "model",
        expName = "main",
        gradientAccumulationSteps = 4,
        batchSize = 8,
        blockSize = 64,
        numberOfLayers = 7,
        numberOfHeads = 2,
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
