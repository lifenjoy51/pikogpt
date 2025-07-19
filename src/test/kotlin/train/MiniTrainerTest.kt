package train

fun main() {
    val config = TrainConfig(
        dataPath = "data/3old",
    )
    val trainer = Trainer(config)
    trainer.train()
}
