package grad

import Value
import kotlin.test.Test

class MlpTest {


    // 사용 예제
    @Test
    fun main() {
        // 3개의 입력, [4, 4, 1]의 은닉층/출력층 구조
        val model = MLP(3, listOf(4, 4, 1))
        println("Model: $model")
        println("Number of parameters: ${model.parameters().size}")

        // 입력 데이터 생성
        val x = listOf(Value(2.0f), Value(3.0f), Value(-1.0f))

        // 순전파
        val output = model(x)
        println("\nForward pass result: $output")

        // 역전파를 위해 output이 Value인 경우 처리
        if (output is Value) {
            output.backward()
            println("\nGradients after backward pass:")
            model.parameters().take(5).forEachIndexed { i, p ->
                println("Parameter $i: $p")
            }
        }

        // 그래디언트 초기화
        model.zeroGrad()
        println("\nAfter zero_grad:")
        model.parameters().take(3).forEachIndexed { i, p ->
            println("Parameter $i grad: ${p.grad}")
        }
    }

    // 훈련 루프 예제
    @Test
    fun trainExample() {
        val model = MLP(2, listOf(16, 16, 1))

        // XOR 문제를 위한 데이터
        val xs = listOf(
            listOf(Value(0.0f), Value(0.0f)),
            listOf(Value(0.0f), Value(1.0f)),
            listOf(Value(1.0f), Value(0.0f)),
            listOf(Value(1.0f), Value(1.0f))
        )
        val ys = listOf(Value(0.0f), Value(1.0f), Value(1.0f), Value(0.0f))

        // 훈련 루프
        for (epoch in 0 until 100) {
            // 순전파
            val ypred = xs.map { x -> model(x) as Value }

            // 손실 계산 (MSE)
            var loss = Value(0.0f)
            for ((yp, y) in ypred.zip(ys)) {
                val diff = yp - y
                loss = loss + diff * diff
            }

            // 역전파
            model.zeroGrad()
            loss.backward()

            // 파라미터 업데이트 (경사 하강법)
            val learningRate = 0.01f
            for (p in model.parameters()) {
                p.data -= learningRate * p.grad
            }

            if (epoch % 10 == 0) {
                println("Epoch $epoch, Loss: ${loss.data}")
            }
        }
    }
}