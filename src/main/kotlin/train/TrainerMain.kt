package train


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
