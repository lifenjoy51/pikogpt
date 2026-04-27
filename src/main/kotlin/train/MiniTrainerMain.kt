package train

fun main() {
    val config = TrainConfig(
        dataPath = "data/az",
        logInterval = 100,
        gradientAccumulationSteps = 1,
        batchSize = 16,
        blockSize = 3,
        embeddingDimension = 4,
        numberOfLayers = 4,
        numberOfHeads = 2,
        learningRate = 1.0e-4f,
    )
    val trainer = ScalarTrainer(config)
    trainer.train()
}
