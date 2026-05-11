package train.experiments

import train.TrainConfig
import train.ScalarTrainer

/**
 * TinyHelen leaner/10M (book+textbook+wiki+conversation) 기반 overnight 학습 엔트리.
 *
 * 이전 overnight(2026-04-23) best val loss = 6.23 @ iter 1200 에서 관측된 문제를 바탕으로
 * 하이퍼파라미터를 재조정한 레시피 A (balanced). 모델 아키텍처(layers=2, heads=3,
 * embd=24, blockSize=48)는 그대로 유지 — 파라미터 수 71,256.
 *
 * 조정 포인트:
 *   - 효과배치 8 → 16 (gradAccum 4→8)  — gradient noise 완화, 곡선 매끄럽게
 *   - dropout 0.1 → 0.0                — tiny + under-train 모델에 해로움
 *   - weightDecay 0.05 → 0.02          — 같은 이유, 약한 신호 억제 방지
 *   - maxIters 1500 → 900              — effective batch 2× 반영, 시간 budget ~8h 근사
 *   - warmupRatio 0.03 → 0.05          — 피크 도달 길이 45 iter
 *   - learningRateDecayRatio 1.0 → 0.85 — 마지막 15%는 min에서 유지 (후반 drift 방지)
 *   - minimumLearningRate 3e-5 → 5e-5  — peak(3e-4)의 1/6, 후반에도 유의미한 업데이트
 *
 * 체크포인트는 검증 손실이 개선될 때마다 `model/{paramCount}/{bestLoss*10}/`에 저장된다.
 */
fun main(args: Array<String>) {
    // 스모크 테스트용: 인자로 maxIters override 가능 (예: runTinyHelenTrain --args="40").
    val maxItersOverride = args.getOrNull(0)?.toIntOrNull()
    val config = TrainConfig(
        dataPath = "data/tinyhelen",
        modelDir = "model",
        // 효과 배치 = batchSize * gradientAccumulationSteps = 16 (이전 8)
        gradientAccumulationSteps = 8,
        batchSize = 2,
        // 모델 아키텍처 (유지)
        blockSize = 48,
        numberOfLayers = 2,
        numberOfHeads = 1,  // ScalarCausalSelfAttention은 single-head만 지원
        embeddingDimension = 24,
        dropout = 0.0f,
        bias = true,
        // 옵티마이저
        learningRate = 3e-4f,
        weightDecay = 0.02f,
        gradClip = 1.0f,
        beta1 = 0.9f,
        beta2 = 0.95f,
        // 스케줄 (레시피 A: 후반 min-LR plateau 확보)
        maxIters = maxItersOverride ?: 900,
        warmupRatio = 0.05f,
        learningRateDecayRatio = 0.85f,
        minimumLearningRate = 5e-5f,
        decayLr = true,
        // 로깅/평가 (no-grad 기반 eval — evalIters=4로 val 추정 안정화)
        evalIntervalRatio = 0.05f,
        evalIters = 4,
        logInterval = 10,
        alwaysSaveCheckpoint = true,
    )
    val trainer = ScalarTrainer(config)
    trainer.train()
}
