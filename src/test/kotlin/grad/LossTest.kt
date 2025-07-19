package grad

import Value
import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class LossTest {

    @Test
    fun test1() {
        // 난수 시드 설정
        Random(1337)

        // 데이터셋 생성
        val (x, y) = makeMoons(100, 0.1f)
        println("데이터셋 생성 완료: ${x.size} 샘플")
        
        // 데이터셋 검증
        assertEquals(100, x.size, "Should have 100 samples")
        assertEquals(100, y.size, "Should have 100 labels")
        assertTrue(y.all { it == -1 || it == 1 }, "Labels should be -1 or 1")
        assertTrue(x.all { it.size == 2 }, "Each sample should have 2 features")

        // 모델 초기화
        val model = MLP(2, listOf(16, 16, 1))
        println("모델: $model")
        println("파라미터 수: ${model.parameters().size}")
        
        // 모델 구조 검증
        assertTrue(model.parameters().size > 0, "Model should have parameters")
        assertEquals(3, model.layers.size, "Should have 3 layers")

        // 손실 계산기 생성
        val lossCalculator = LossCalculator(model, x, y)

        // 초기 손실 계산
        val (initialLoss, initialAcc) = lossCalculator.loss()
        println("초기 손실: ${initialLoss.scalarValue}, 정확도: ${initialAcc * 100}%")
        
        // 초기 손실이 합리적인 범위인지 확인
        assertTrue(initialLoss.scalarValue > 0, "Initial loss should be positive")
        assertTrue(initialAcc >= 0.0f && initialAcc <= 1.0f, "Initial accuracy should be between 0 and 1")

        // 최적화 (학습)
        for (k in 0 until 100) {
            // 순전파
            val (totalLoss, acc) = lossCalculator.loss()

            // 역전파
            model.zeroGrad()
            totalLoss.backward()

            // 파라미터 업데이트 (SGD)
            val learningRate: Float = 1.0f - 0.9f * k / 100
            for (p in model.parameters()) {
                p.scalarValue -= learningRate * p.gradient
            }

            // 10 스텝마다 결과 출력
            if (k % 10 == 0) {
                println("스텝 $k - 손실: ${totalLoss.scalarValue}, 정확도: ${acc * 100}%")
            }
        }

        // 최종 평가
        val (finalLoss, finalAcc) = lossCalculator.loss()
        println("\n학습 완료!")
        println("최종 손실: ${finalLoss.scalarValue}")
        println("최종 정확도: ${finalAcc * 100}%")
        
        // 학습 효과 검증
        assertTrue(finalLoss.scalarValue < initialLoss.scalarValue, "Loss should decrease after training (initial: ${initialLoss.scalarValue}, final: ${finalLoss.scalarValue})")
        assertTrue(finalAcc > initialAcc, "Accuracy should improve after training (initial: $initialAcc, final: $finalAcc)")
        assertTrue(finalAcc > 0.5f, "Final accuracy should be better than random guessing")

        // 예측 테스트
        println("\n예측 테스트:")
        val testPoints = arrayOf(
            floatArrayOf(0.5f, 0.5f),
            floatArrayOf(-0.5f, 0.5f),
            floatArrayOf(0.5f, -0.5f),
            floatArrayOf(-0.5f, -0.5f)
        )

        val predictions = mutableListOf<Int>()
        for (point in testPoints) {
            val input = point.map { Value(it) }
            val output = model(input) as Value
            val prediction = if (output.scalarValue > 0) 1 else -1
            predictions.add(prediction)
            println("입력: (${point[0]}, ${point[1]}) -> 예측: $prediction (점수: ${output.scalarValue})")
            
            // 출력이 Value 타입인지 확인
            assertTrue(output is Value, "Model output should be a Value")
        }
        
        // 예측이 올바른 범위에 있는지 확인
        assertTrue(predictions.all { it == -1 || it == 1 }, "All predictions should be -1 or 1")
        assertEquals(4, predictions.size, "Should have 4 predictions")
    }

    @Test
    fun test2() {
        // 배치 학습 예제
        val (x, y) = makeMoons(100, 0.1f)
        val model = MLP(2, listOf(16, 16, 1))
        val lossCalculator = LossCalculator(model, x, y)

        println("배치 학습 시작...")
        
        // 초기 성능 저장
        val (initialLoss, initialAcc) = lossCalculator.loss()
        assertTrue(initialLoss.scalarValue > 0, "Initial loss should be positive")
        assertTrue(initialAcc in 0.0f..1.0f, "Initial accuracy should be between 0 and 1")

        for (epoch in 0 until 500) {
            // 배치 크기 32로 학습
            val (totalLoss, acc) = lossCalculator.loss(32)

            model.zeroGrad()
            totalLoss.backward()

            val learningRate = 0.1f
            for (p in model.parameters()) {
                p.scalarValue -= learningRate * p.gradient
            }

            if (epoch % 10 == 0) {
                val (evalLoss, evalAcc) = lossCalculator.loss()
                println("에폭 $epoch - 손실: ${evalLoss.scalarValue}, 정확도: ${evalAcc * 100}%")
            }
        }
        
        // 최종 평가
        val (finalLoss, finalAcc) = lossCalculator.loss()
        
        // 배치 학습 효과 검증
        assertTrue(finalLoss.scalarValue < initialLoss.scalarValue, "Batch training should reduce loss")
        assertTrue(finalAcc >= initialAcc, "Batch training should maintain or improve accuracy")
        assertTrue(finalAcc > 0.6f, "Final accuracy should be reasonable for batch training")

        // 예측 테스트
        println("\n예측 테스트:")
        val testPoints = arrayOf(
            floatArrayOf(0.5f, 0.5f),
            floatArrayOf(-0.5f, 0.5f),
            floatArrayOf(0.5f, -0.5f),
            floatArrayOf(-0.5f, -0.5f)
        )

        val batchPredictions = mutableListOf<Int>()
        for (point in testPoints) {
            val input = point.map { Value(it) }
            val output = model(input) as Value
            val prediction = if (output.scalarValue > 0) 1 else -1
            batchPredictions.add(prediction)
            println("입력: (${point[0]}, ${point[1]}) -> 예측: $prediction (점수: ${output.scalarValue})")
            
            // 출력 검증
            assertTrue(output is Value, "Model output should be a Value")
        }
        
        // 배치 학습 예측 검증
        assertTrue(batchPredictions.all { it == -1 || it == 1 }, "All predictions should be -1 or 1")
        assertEquals(4, batchPredictions.size, "Should have 4 predictions")
    }

    // 데이터 생성을 위한 함수 (make_moons와 유사)
    private fun makeMoons(nSamples: Int, noise: Float = 0.1f): Pair<Array<FloatArray>, IntArray> {
        val x = Array(nSamples) { FloatArray(2) }
        val y = IntArray(nSamples)

        val random = Random(1337)

        for (i in 0 until nSamples) {
            val angle = Math.PI * i / (nSamples / 2)
            if (i < nSamples / 2) {
                // 첫 번째 반달
                x[i][0] = cos(angle).toFloat() + random.nextFloat() * noise
                x[i][1] = sin(angle).toFloat() + random.nextFloat() * noise
                y[i] = -1
            } else {
                // 두 번째 반달
                x[i][0] = 1 - cos(angle).toFloat() + random.nextFloat() * noise
                x[i][1] = 1 - sin(angle).toFloat() - 0.5f + random.nextFloat() * noise
                y[i] = 1
            }
        }

        return Pair(x, y)
    }
    
    @Test
    fun testMakeMoons() {
        val (x, y) = makeMoons(50, 0.05f)
        
        // 데이터 생성 검증
        assertEquals(50, x.size, "Should generate correct number of samples")
        assertEquals(50, y.size, "Should generate correct number of labels")
        
        // 특성 검증
        assertTrue(x.all { it.size == 2 }, "Each sample should have 2 features")
        assertTrue(y.all { it == -1 || it == 1 }, "Labels should be -1 or 1")
        
        // 클래스 분포 검증 (대략 반반)
        val negativeCount = y.count { it == -1 }
        val positiveCount = y.count { it == 1 }
        assertEquals(25, negativeCount, "Should have 25 negative samples")
        assertEquals(25, positiveCount, "Should have 25 positive samples")
    }

}