package train.experiments

import turbo.TurboTrainConfig
import turbo.TurboTrainer

/**
 * ccmc-lemma-v1024 — lemma_sentences.txt만 사용한 vocab=1024 짧은-문장 코퍼스.
 *
 * 데이터 통계:
 *   - 78,958 sentences, train ~1.33M tokens / val ~70k tokens
 *   - 평균 길이 16.7 tokens, 중앙값 16, P99=32, max 130
 *   - 모든 record가 `<|bos|> ... <|eos|>`로 래핑 (BOS/EOS 각 5.65%)
 *
 * 짧은 문장 특성에 맞춰 blockSize=32 (P99 cover 99.01%), maxSeqLen=32.
 *
 * 모델 (트랜스포머 비중 ≈ 65%, ~97k params):
 *   embedDim=32, numHeads=2 (head_dim=16, SIMD lane 친화), numLayers=5
 *   - Token emb (1024×32): 32,768 (33.65%)
 *   - Pos emb (32×32): 1,024 (1.05%)
 *   - 5× Transformer block (12,704 each): 63,520 (65.23%)
 *     · Attn 4d²+4d = 4,224
 *     · 2 LayerNorm: 128
 *     · MLP GELU 8d²+5d: 8,352
 *   - Final LN: 64
 *   - LM head: 0 (tied)
 *   합계: 97,376
 *
 * 비교용 학습량 매칭: 10,000 iter (CcmcAllCompareElfTrainTurbo와 동일 스케줄).
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TurboTrainConfig(
        dataPath = "data/ccmc-lemma-v1024",
        modelDir = "model",
        expName = "main",
        gradientAccumulationSteps = 4,
        batchSize = 8,
        blockSize = 32,
        numberOfLayers = 5,
        numberOfHeads = 2,
        embeddingDimension = 32,
        dropout = 0.1f,
        bias = true,
        learningRate = 3e-4f,
        weightDecay = 0.01f,
        gradClip = 1.0f,
        beta1 = 0.9f,
        beta2 = 0.95f,
        maxIters = maxItersOverride ?: 10000,
        warmupRatio = 0.05f,
        learningRateDecayRatio = 0.95f,
        minimumLearningRate = 1e-5f,
        decayLr = true,
        evalIntervalRatio = 0.05f,
        evalIters = 10,
        logInterval = 100,
        alwaysSaveCheckpoint = true,
        earlyStopPatience = 0,
        initFrom = if (resume) "resume" else "scratch",
        modelCheckpointDir = null,
        tieWeights = true,
    )

    TurboTrainer(config).train()
}
