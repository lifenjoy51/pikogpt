package train


// 메인 함수 8192*3 == 24576
fun main() {
    //train_3old()
    train_3old_fast()
}

fun train_3old_fast() {
    val config = TrainConfig(
        dataPath = "data/3old",
        nLayer = 1,
        nHead = 1,
        nEmbd = 4,
        maxIters = 100,
        evalIters = 100,
        alwaysSaveCheckpoint = false
    )
    val trainer = Trainer(config)
    trainer.train()
}

fun train_3old() {
    val config = TrainConfig(
        dataPath = "data/3old",
    )
    val trainer = Trainer(config)
    trainer.train()
}

fun resume(){
    val config = TrainConfig(
        initFrom = "resume",
        subDir = "1752368702662/47"
    )
    val trainer = Trainer(config)
    trainer.train()

}

