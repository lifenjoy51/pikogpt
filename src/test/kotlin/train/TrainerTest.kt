package train

import org.junit.jupiter.api.Assertions.*

class TrainerTest {


    // 메인 함수
    fun main() {
        val config = TrainConfig()
        val trainer = Trainer(config)
        trainer.train()
    }
}