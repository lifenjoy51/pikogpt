package train


// 메인 함수 8192*3 == 24576
fun main() {
    train_3old()
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

