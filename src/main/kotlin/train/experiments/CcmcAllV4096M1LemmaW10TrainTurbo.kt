package train.experiments

import turbo.TurboTrainConfig
import turbo.TurboTrainer

/**
 * ccmc-all-v4096-v2 (vocab=4096, 7파일 corpus, lemma 가중치 0.1) ~1.07M tied 모델 turbo — 120000 iter.
 *
 * v9(=CcmcAllV4096M1TrainTurbo) 대비 변경:
 *   - dataPath: ccmc-all-v4096-v1 → ccmc-all-v4096-v2 (lemma/other 분리 BPE)
 *   - lemmaSamplingRatio = 0.1f (lemma stream을 batch sequence별 10% 확률로만 sampling)
 *   - val.bin은 lemma+other 자연 비율 유지 (평가 신호 보존)
 *   - 나머지 hyperparam 전부 v9와 동일
 *
 * 목적: EOS 학습 빈도 균형 — chunk당 평균 EOS 1.75 → ~1.18 (non-lemma 1.10 근접).
 *       단답 prior 완화로 장문 generation 능력 강화 기대.
 *
 * 학습 데이터:
 *   - data/ccmc-all-v4096-v2/train_other.bin: 5.17M tokens (primary, p=0.9)
 *     · stories + dialogues + wiki + cause_seq + chained + counting
 *   - data/ccmc-all-v4096-v2/train_lemma.bin: 1.08M tokens (secondary, p=0.1)
 *     · lemma_sentences (모든 라인 보존, sampling 확률만 1/10로)
 *   - data/ccmc-all-v4096-v2/val_other.bin: 274k tokens, val_lemma.bin: 57k tokens
 *     · 평가도 동일 weighted source (secondary p=0.1) → train/eval 분포 정확히 일치
 *
 * 평가: 매 1200 iter (evalIntervalRatio=0.01 → 최대 100 ckpt)
 * Early stop: patience=20.
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TurboTrainConfig(
        dataPath = "data/ccmc-all-v4096-v2",
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
        lemmaSamplingRatio = 0.1f,
    )

    TurboTrainer(config).train()
}
