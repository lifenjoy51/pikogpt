package train.experiments

import train.TrainConfig
import train.ScalarTrainer

/**
 * **conv-mix-clean-a510 + 773k tied** — 데이터 정제 후 재학습.
 *
 * 직전 ConvMixA510M773TrainVec에서 chat 응답에 `**child**` 같은 markdown emphasis가
 * 그대로 출력됨. 원인: 데이터 전처리 정규식 `\*\*[^*]+\*\*:\s*`이 콜론 있는 speaker 마커만
 * 처리해 본문 내 emphasis(콜론 없음, 약 3,474개의 "**Classmate**", "**Child**" 등)가
 * 그대로 남음. 모델이 그 패턴을 학습.
 *
 * Fix:
 *   1. `\*\*[^*]+\*\*:\s*` → `<|turn|>` (기존, speaker 마커)
 *   2. `\*\*([^*]+)\*\*` → `\1` (남은 emphasis는 안의 텍스트만 — `**Classmate**` → `Classmate`)
 *
 * 데이터: `data/conv-mix-clean-a510/`
 *   - 동일 소스 (TinyHelen + age-5 + age-10)
 *   - emphasis 마커 정리 후 약 18M tokens (이전 18.9M과 거의 동일)
 *
 * 모델·하이퍼파라미터: ConvMixA510M773TrainVec와 100% 동일 (직접 비교)
 *   - layers 6 · dim 96 · heads 3 · head dim 32 · block 64 · tied = 773k params
 *   - LR 3e-4 / dropout 0.05 / wd 0.05 / LS 0.05 / gradClip 1.0
 *   - maxIters 12000
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TrainConfig(
        dataPath = "data/conv-mix-clean-a510",
        modelDir = "model",
        gradientAccumulationSteps = 32,
        batchSize = 2,
        blockSize = 64,
        numberOfLayers = 6,
        numberOfHeads = 3,
        embeddingDimension = 96,
        dropout = 0.05f,
        bias = true,
        tieWeights = true,
        learningRate = 3e-4f,
        weightDecay = 0.05f,
        labelSmoothing = 0.05f,
        gradClip = 1.0f,
        beta1 = 0.9f,
        beta2 = 0.95f,
        maxIters = maxItersOverride ?: 12000,
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

    vec.VecTrainer(config).train()
}
