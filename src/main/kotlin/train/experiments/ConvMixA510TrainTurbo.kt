package train.experiments

import turbo.TurboTrainConfig
import train.ScalarTrainer

/**
 * **conv-mix-turn-noq-a510** — conv-mix 확장: TinyDialogues age-5 + age-10 + TinyHelen conv.
 *
 * 직전 conv-mix-turn-noq run(9.5M tokens, val 2.96)에서 데이터가 진짜 한계라고 판단.
 * age-10 추가로 18M tokens 확보 → 432k 모델 기준 Chinchilla **41×** (권고 20×의 2배).
 *
 * 데이터: `data/conv-mix-turn-noq-a510/`
 *   - TinyHelen conv 191 docs (영화 스크립트, 따옴표 형식)
 *   - TinyDialogues age-5 28,188 docs (CDS, 5세 어휘)
 *   - TinyDialogues age-10 23,041 docs (10세 어휘 — Child 응답 더 풍부)
 *   - 모두 `**Speaker**:` 제거, outer `"` 제거, 발화 사이 `<|turn|>` 삽입
 *   - train ~18M tokens / val ~3M tokens
 *
 * 모델: 432k params (이전과 동일 — 데이터 증량 효과 검증)
 *   - layers 6 · dim 64 · heads 2 · head dim 32 · block 64
 *
 * 하이퍼파라미터: 이전과 동일 (직접 비교 목적)
 *   - batch × accum = 2 × 32 = 64
 *   - LR 3e-4 / dropout 0.05 / wd 0.05 / LS 0.05 / gradClip 1.0
 *   - maxIters 8000 (Chinchilla 41× → ~17× per token, 충분)
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TurboTrainConfig(
        dataPath = "data/conv-mix-turn-noq-a510",
        modelDir = "model",
        expName = "a510",
        gradientAccumulationSteps = 32,
        batchSize = 2,
        blockSize = 64,
        numberOfLayers = 6,
        numberOfHeads = 2,
        embeddingDimension = 64,
        dropout = 0.05f,
        bias = true,
        learningRate = 3e-4f,
        weightDecay = 0.05f,
        labelSmoothing = 0.05f,
        gradClip = 1.0f,
        tieWeights = false,  // 이전 conv-mix-turn-noq 432k 베이스와 직접 비교 위해 untied 유지
        beta1 = 0.9f,
        beta2 = 0.95f,
        maxIters = maxItersOverride ?: 8000,
        warmupRatio = 0.03f,
        learningRateDecayRatio = 0.95f,
        minimumLearningRate = 3e-5f,
        decayLr = true,
        evalIntervalRatio = 0.05f,
        evalIters = 100,
        logInterval = 100,
        alwaysSaveCheckpoint = false,
        initFrom = if (resume) "resume" else "scratch",
    )

    turbo.TurboTrainer(config).train()
}
