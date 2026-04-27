package train.experiments

import train.TrainConfig
import train.ScalarTrainer

/**
 * TinyHelen 코퍼스 overnight 학습 — 벡터 백엔드 (`vec.Trainer`) 용 엔트리.
 *
 * 스칼라 `TinyHelenTrain`과 같은 데이터(`data/tinyhelen`)를 쓰지만
 * 아키텍처는 **약 1M 파라미터**로 확대:
 *   - numberOfLayers        : 4
 *   - embeddingDimension    : 128
 *   - numberOfHeads         : 4  (head dim = 32)
 *   - MLP hidden (FFN)      : 512 (= 4 * embedDim, `vec.layer.MLP`가 자동 설정)
 *   - blockSize             : 48  (컨텍스트는 유지)
 *
 * 대략적 파라미터 수:
 *   tok_emb(1000×128) + pos_emb(48×128) + 4×block(≈198K) + final_ln + lm_head(128×1000)
 *   = 128K + 6K + 793K + 256 + 128K ≈ **1.05M**
 *
 * 스칼라 autodiff로는 이 규모가 비현실적(≈2-4분/iter × 1500 iter). 벡터 백엔드로는
 * iter 훨씬 짧음 (FloatArray loop 기반). 체크포인트는 `${modelDir}/vec/{params}/{loss*10}/`.
 *
 * CLI 인자:
 *   - 숫자가 주어지면 maxIters override (`--args="20"` smoke)
 *   - `"resume"` 키워드가 있으면 `model/vec/${paramCount}/` 아래 체크포인트 중
 *     `iterationNumber` 최대인 것에서 이어서 학습 (`--args="resume"` 또는 `--args="20 resume"`)
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TrainConfig(
        dataPath = "data/tinyhelen",
        modelDir = "model",
        // 효과 배치 = batch * accum = 32 (노이즈 감소)
        gradientAccumulationSteps = 16,
        batchSize = 2,
        blockSize = 64,
        // 모델 (≈1M 파라미터)
        numberOfLayers = 4,
        numberOfHeads = 4,
        embeddingDimension = 128,
        dropout = 0.0f,
        bias = true,
        // 옵티마이저 (recipe A 기반)
        learningRate = 3e-4f,
        weightDecay = 0.02f,
        gradClip = 1.0f,
        beta1 = 0.9f,
        beta2 = 0.95f,
        // 스케줄 (long run 용)
        //   - warmup 3% (0.03 * 10000 = 300 iter)
        //   - cosine decay 95%까지 (iter 9500), 마지막 5%는 min LR plateau
        //   - min LR 3e-5: peak(3e-4)의 1/10, 후반에도 충분히 작은 업데이트 허용
        maxIters = maxItersOverride ?: 10000,
        warmupRatio = 0.03f,
        learningRateDecayRatio = 0.95f,
        minimumLearningRate = 3e-5f,
        decayLr = true,
        // 평가/로깅
        //   10000 iter × 0.05 = 500 iter마다 eval → 20회
        //   log는 100 iter마다 (긴 run에 로그 과다 방지)
        evalIntervalRatio = 0.05f,
        evalIters = 16,
        logInterval = 100,
        alwaysSaveCheckpoint = true,
        initFrom = if (resume) "resume" else "scratch",
    )

    vec.VecTrainer(config).train()
}
