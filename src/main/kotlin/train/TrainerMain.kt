package train

/**
 * 옛 학습 진입점 — 현재 `data/simple` 데이터에 의존하지만 그 폴더가 저장소에 없어
 * 그대로는 동작하지 않습니다.
 *
 * **Scalar 백엔드 quickstart는 [MiniTrainerMain]을 사용하세요.** alphabet 데이터로
 * 코드 변경 없이 즉시 학습→샘플링이 도는 정비된 진입점입니다. 자세한 가이드는
 * `docs/scalar-quickstart.md`.
 *
 * 이 파일은 사용자 정의 학습을 직접 짜고 싶을 때의 참조 템플릿으로 남겨둡니다.
 * 새로 사용하려면 `dataPath`를 본인의 데이터셋으로 바꾸고 모델/하이퍼파라미터를 조정하세요.
 */
fun main() {
    train()
//    resume()
}

val config = TrainConfig(
    dataPath = "data/simple",
    gradientAccumulationSteps = 4,
    batchSize = 2,
    blockSize = 48,
    numberOfLayers = 2,
    numberOfHeads = 2,
    embeddingDimension = 16,
    maxIters = 50,
    warmupRatio = 0.1f,
    learningRateDecayRatio = 1.0f,
    evalIntervalRatio = 0.2f,
    evalIters = 1,
)

fun train() {
    val trainer = ScalarTrainer(config)
    trainer.train()
}

fun resume(){
    val config = TrainConfig(
        initFrom = "resume",
        dataPath = "data/1k",
        modelCheckpointDir = "model/78096/36",
        gradientAccumulationSteps = 1,
        batchSize = 8,
        blockSize = 48,
        numberOfLayers = 4,
        numberOfHeads = 6,
        embeddingDimension = 24,
        learningRate = 5.0e-5f,
    )
    val trainer = ScalarTrainer(config)
    trainer.train()

}
