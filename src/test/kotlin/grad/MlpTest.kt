package grad

import Value
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

class MlpTest {


    // 사용 예제
    @Test
    fun testMlp() {
        // 3개의 입력, [4, 4, 1]의 은닉층/출력층 구조
        val model = MLP(3, listOf(4, 4, 1))
        println("Model: $model")
        println("Number of parameters: ${model.parameters().size}")
        
        // 모델이 올바르게 생성되었는지 확인
        assertTrue(model.parameters().size > 0, "Should have parameters")

        // 입력 데이터 생성
        val x = listOf(Value(2.0f), Value(3.0f), Value(-1.0f))

        // 순전파
        val output = model(x)
        println("\nForward pass result: $output")
        
        // 출력이 Value 타입인지 확인 (마지막 층의 출력은 단일 값)
        assertTrue(output is Value, "Output should be a Value")

        // 역전파를 위해 output이 Value인 경우 처리
        if (output is Value) {
            output.backward()
            println("\nGradients after backward pass:")
            model.parameters().take(5).forEachIndexed { i, p ->
                println("Parameter $i: $p")
            }
            
            // 역전파 후 그래디언트가 계산되었는지 확인
            val hasNonZeroGrad = model.parameters().any { it.gradient != 0.0f }
            assertTrue(hasNonZeroGrad, "Some gradients should be non-zero after backward pass")
        }

        // 그래디언트 초기화
        model.zeroGrad()
        println("\nAfter zero_grad:")
        model.parameters().take(3).forEachIndexed { i, p ->
            println("Parameter $i grad: ${p.gradient}")
        }
        
        // 모든 그래디언트가 0으로 초기화되었는지 확인
        assertTrue(model.parameters().all { it.gradient == 0.0f }, "All gradients should be zero after zeroGrad")
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

        // 초기 손실 저장
        val initialLoss = run {
            val ypred = xs.map { x -> model(x) as Value }
            var loss = Value(0.0f)
            for ((yp, y) in ypred.zip(ys)) {
                val diff = yp - y
                loss = loss + diff * diff
            }
            loss.scalarValue
        }
        
        // 훈련 루프
        var finalLoss = 0.0f
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
                p.scalarValue -= learningRate * p.gradient
            }

            if (epoch % 10 == 0) {
                println("Epoch $epoch, Loss: ${loss.scalarValue}")
            }
            
            if (epoch == 99) {
                finalLoss = loss.scalarValue
            }
        }
        
        // 훈련 후 손실이 감소했는지 확인
        assertTrue(finalLoss < initialLoss, "Loss should decrease after training (initial: $initialLoss, final: $finalLoss)")
        
        // XOR 문제 테스트 - 훈련된 모델로 예측
        val testPredictions = xs.map { x -> (model(x) as Value).scalarValue }
        println("\nFinal predictions: $testPredictions")
        
        // 예측값이 타겟에 어느 정도 가까워졌는지 확인 (완벽하지 않을 수 있음)
        val expectedOutputs = listOf(0.0f, 1.0f, 1.0f, 0.0f)
        for (i in testPredictions.indices) {
            val prediction = testPredictions[i]
            val expected = expectedOutputs[i]
            assertTrue(abs(prediction - expected) < 0.8f, 
                "Prediction $i should be closer to expected value (predicted: $prediction, expected: $expected)")
        }
    }
}