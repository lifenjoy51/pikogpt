package grad

import Value
import kotlin.math.*
import kotlin.random.Random

// 데이터 생성을 위한 함수 (make_moons와 유사)
fun makeMoons(nSamples: Int, noise: Double = 0.1): Pair<Array<DoubleArray>, IntArray> {
    val x = Array(nSamples) { DoubleArray(2) }
    val y = IntArray(nSamples)

    val random = Random(1337)

    for (i in 0 until nSamples) {
        val angle = Math.PI * i / (nSamples / 2)
        if (i < nSamples / 2) {
            // 첫 번째 반달
            x[i][0] = cos(angle) + random.nextDouble() * noise
            x[i][1] = sin(angle) + random.nextDouble() * noise
            y[i] = -1
        } else {
            // 두 번째 반달
            x[i][0] = 1 - cos(angle) + random.nextDouble() * noise
            x[i][1] = 1 - sin(angle) - 0.5 + random.nextDouble() * noise
            y[i] = 1
        }
    }

    return Pair(x, y)
}


// 메인 데모 함수
fun main1() {
    // 난수 시드 설정
    Random(1337)

    // 데이터셋 생성
    val (x, y) = makeMoons(100, 0.1)
    println("데이터셋 생성 완료: ${x.size} 샘플")

    // 모델 초기화
    val model = MLP(2, listOf(16, 16, 1))
    println("모델: $model")
    println("파라미터 수: ${model.parameters().size}")

    // 손실 계산기 생성
    val lossCalculator = LossCalculator(model, x, y)

    // 초기 손실 계산
    val (initialLoss, initialAcc) = lossCalculator.loss()
    println("초기 손실: ${initialLoss.data}, 정확도: ${initialAcc * 100}%")

    // 최적화 (학습)
    for (k in 0 until 100) {
        // 순전파
        val (totalLoss, acc) = lossCalculator.loss()

        // 역전파
        model.zeroGrad()
        totalLoss.backward()

        // 파라미터 업데이트 (SGD)
        val learningRate = 1.0 - 0.9 * k / 100
        for (p in model.parameters()) {
            p.data -= learningRate * p.grad
        }

        // 10 스텝마다 결과 출력
        if (k % 10 == 0) {
            println("스텝 $k - 손실: ${totalLoss.data}, 정확도: ${acc * 100}%")
        }
    }

    // 최종 평가
    val (finalLoss, finalAcc) = lossCalculator.loss()
    println("\n학습 완료!")
    println("최종 손실: ${finalLoss.data}")
    println("최종 정확도: ${finalAcc * 100}%")

    // 예측 테스트
    println("\n예측 테스트:")
    val testPoints = arrayOf(
        doubleArrayOf(0.5, 0.5),
        doubleArrayOf(-0.5, 0.5),
        doubleArrayOf(0.5, -0.5),
        doubleArrayOf(-0.5, -0.5)
    )

    for (point in testPoints) {
        val input = point.map { Value(it) }
        val output = model(input) as Value
        val prediction = if (output.data > 0) 1 else -1
        println("입력: (${point[0]}, ${point[1]}) -> 예측: $prediction (점수: ${output.data})")
    }
}

// 배치 학습을 위한 확장 함수
fun main2() {
    // 배치 학습 예제
    val (x, y) = makeMoons(100, 0.1)
    val model = MLP(2, listOf(16, 16, 1))
    val lossCalculator = LossCalculator(model, x, y)

    println("배치 학습 시작...")

    for (epoch in 0 until 500) {
        // 배치 크기 32로 학습
        val (totalLoss, acc) = lossCalculator.loss(32)

        model.zeroGrad()
        totalLoss.backward()

        val learningRate = 0.1
        for (p in model.parameters()) {
            p.data -= learningRate * p.grad
        }

        if (epoch % 10 == 0) {
            val (evalLoss, evalAcc) = lossCalculator.loss()
            println("에폭 $epoch - 손실: ${evalLoss.data}, 정확도: ${evalAcc * 100}%")
        }
    }

    // 예측 테스트
    println("\n예측 테스트:")
    val testPoints = arrayOf(
        doubleArrayOf(0.5, 0.5),
        doubleArrayOf(-0.5, 0.5),
        doubleArrayOf(0.5, -0.5),
        doubleArrayOf(-0.5, -0.5)
    )

    for (point in testPoints) {
        val input = point.map { Value(it) }
        val output = model(input) as Value
        val prediction = if (output.data > 0) 1 else -1
        println("입력: (${point[0]}, ${point[1]}) -> 예측: $prediction (점수: ${output.data})")
    }
}

fun main() {
    //main1()
    main2()
}
