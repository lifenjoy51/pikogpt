package train


fun main() {
    train()
//    resume()
}

val config = TrainConfig(
    dataPath = "data/simple",
    gradientAccumulationSteps = 8,
    batchSize = 4,
    blockSize = 64,
    numberOfLayers = 6,
    numberOfHeads = 4,
    embeddingDimension = 32,
)

fun train() {
    val trainer = Trainer(config)
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
    val trainer = Trainer(config)
    trainer.train()

}

