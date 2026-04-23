package train

/**
 * TinyHelen leaner/10M (book+textbook+wiki+conversation) 기반 overnight 학습 엔트리.
 *
 * 스칼라 autodiff 성능 한계를 감안해 작은 모델로 오래 돌리는 방향으로 설정함.
 * 체크포인트는 검증 손실이 개선될 때마다 `model/{paramCount}/{bestLoss*10}/`에 저장된다.
 */
fun main() {
    val config = TrainConfig(
        dataPath = "data/tinyhelen",
        modelDir = "model",
        // 효과 배치 = batchSize * gradientAccumulationSteps = 8
        gradientAccumulationSteps = 4,
        batchSize = 2,
        blockSize = 48,
        // 모델 (이전 세션 OOM 회피를 위해 보수적으로)
        numberOfLayers = 2,
        numberOfHeads = 3,
        embeddingDimension = 24, // 24 / 3 heads = 8 dim/head
        dropout = 0.1f,
        bias = true,
        // 옵티마이저
        learningRate = 3e-4f,
        weightDecay = 0.05f,
        gradClip = 1.0f,
        beta1 = 0.9f,
        beta2 = 0.95f,
        // 스케줄 (밤새 학습 기준)
        maxIters = 1500,
        warmupRatio = 0.03f,
        learningRateDecayRatio = 1.0f,
        minimumLearningRate = 3e-5f,
        decayLr = true,
        // 로깅/평가 (evalIters=1 — estimateLoss 병렬도 = 2(train+val), 8GB 힙 한계)
        evalIntervalRatio = 0.05f,
        evalIters = 1,
        logInterval = 10,
        alwaysSaveCheckpoint = true,
    )
    val trainer = Trainer(config)
    trainer.train()
}
