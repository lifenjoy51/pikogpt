package grad

import Value
import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class LossTest {

    /**
     * make-moons 분류 학습 — `MicrogradLossCalculator`로 loss + accuracy를 한번에 받는 흐름.
     *
     * 한 step에서 다음 4단계가 반복됩니다:
     *   Step 1 — forward pass + loss computation: lossCalculator.loss()가 모든 샘플의 forward를
     *            돌리며 동시에 hinge loss를 합산하고 정확도를 계산.
     *   Step 2 — backward propagation: 누적 loss 노드에서 chain rule.
     *   Step 3 — parameter update: SGD step (learning rate decay 적용).
     *
     * 학습이 진행되면 loss는 줄고 정확도는 올라갑니다.
     */
    @Test
    fun test1() {
        // 난수 시드 설정
        Random(1337)

        // 데이터셋 생성
        val (x, y) = makeMoons(50, 0.1f)
        println("데이터셋 생성 완료: ${x.size} 샘플")

        // 데이터셋 검증
        assertEquals(50, x.size, "Should have 50 samples")
        assertEquals(50, y.size, "Should have 50 labels")
        assertTrue(y.all { it == -1 || it == 1 }, "Labels should be -1 or 1")
        assertTrue(x.all { it.size == 2 }, "Each sample should have 2 features")

        // 모델 초기화
        val model = MicrogradMLP(2, listOf(16, 16, 1))
        println("모델: $model")
        println("파라미터 수: ${model.parameters().size}")

        // 모델 구조 검증
        assertTrue(model.parameters().isNotEmpty(), "Model should have parameters")
        assertEquals(3, model.layers.size, "Should have 3 layers")

        // 손실 계산기 생성
        val lossCalculator = MicrogradLossCalculator(model, x, y)

        // 초기 손실 계산 (학습 효과 비교용)
        val (initialLoss, initialAcc) = lossCalculator.loss()
        println("초기 손실: ${initialLoss.scalarValue}, 정확도: ${initialAcc * 100}%")

        // 초기 손실이 합리적인 범위인지 확인
        assertTrue(initialLoss.scalarValue > 0, "Initial loss should be positive")
        assertTrue(initialAcc in 0.0f..1.0f, "Initial accuracy should be between 0 and 1")

        // 학습 루프
        val totalSteps = 40
        for (k in 0 until totalSteps) {
            // Step 1 — forward pass + loss computation (lossCalculator가 둘을 한 번에)
            val (totalLoss, acc) = lossCalculator.loss()

            // Step 2 — backward propagation (zeroGrad로 이전 gradient 청소 후 chain rule)
            model.zeroGrad()
            totalLoss.backward()

            // Step 3 — parameter update (SGD with learning rate decay).
            // lr가 1.0에서 0.1로 점차 감소 — 초기에 큰 step, 후반에 안정 수렴.
            val learningRate: Float = 1.0f - 0.9f * k / totalSteps
            for (p in model.parameters()) {
                p.scalarValue -= learningRate * p.gradient
            }

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
        }
        
        // 예측이 올바른 범위에 있는지 확인
        assertTrue(predictions.all { it == -1 || it == 1 }, "All predictions should be -1 or 1")
        assertEquals(4, predictions.size, "Should have 4 predictions")
    }

    /**
     * 배치 학습 예제 — `lossCalculator.loss(batchSize)`로 mini-batch SGD.
     *
     * test1과 같은 4단계 (forward+loss / backward / update)지만 매 epoch마다 60개 중 20개 샘플만
     * 사용. mini-batch는 학습 신호에 약간의 노이즈를 주어 일반화에 도움.
     */
    @Test
    fun test2() {
        // 배치 학습 예제
        val (x, y) = makeMoons(60, 0.1f)
        val model = MicrogradMLP(2, listOf(16, 16, 1))
        val lossCalculator = MicrogradLossCalculator(model, x, y)

        println("배치 학습 시작...")

        // 초기 성능 저장
        val (initialLoss, initialAcc) = lossCalculator.loss()
        assertTrue(initialLoss.scalarValue > 0, "Initial loss should be positive")
        assertTrue(initialAcc in 0.0f..1.0f, "Initial accuracy should be between 0 and 1")

        val totalEpochs = 80
        for (epoch in 0 until totalEpochs) {
            // Step 1 — forward + loss (mini-batch 20개)
            val (totalLoss, _) = lossCalculator.loss(20)

            // Step 2 — backward
            model.zeroGrad()
            totalLoss.backward()

            // Step 3 — parameter update (constant lr)
            val learningRate = 0.1f
            for (p in model.parameters()) {
                p.scalarValue -= learningRate * p.gradient
            }

            // 평가는 전체 데이터로
            if (epoch % 20 == 0) {
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