package grad

import Value

// 손실 함수 구현
class LossCalculator(
    private val model: MLP,
    private val x: Array<DoubleArray>,
    private val y: IntArray
) {
    fun loss(batchSize: Int? = null): Pair<Value, Double> {
        // 배치 선택
        val (xb, yb) = if (batchSize == null) {
            Pair(x, y)
        } else {
            val indices = (0 until x.size).shuffled().take(batchSize)
            val xBatch = indices.map { x[it] }.toTypedArray()
            val yBatch = indices.map { y[it] }.toIntArray()
            Pair(xBatch, yBatch)
        }

        // 입력 데이터를 Value 객체로 변환
        val inputs = xb.map { xrow ->
            xrow.map { Value(it) }
        }

        // 모델 순전파
        val scores = inputs.map { input ->
            model(input) as Value
        }

        // SVM "max-margin" 손실
        val losses = scores.zip(yb.toList()) { score, yi ->
            (Value(1.0) + Value(-yi.toDouble()) * score).relu()
        }

        val dataLoss = losses.reduce { acc, loss -> acc + loss } * Value(1.0 / losses.size)

        // L2 정규화
        val alpha = 1e-4
        val regLoss = model.parameters()
            .map { p -> p * p }
            .reduce { acc, p2 -> acc + p2 } * Value(alpha)

        val totalLoss = dataLoss + regLoss

        // 정확도 계산
        val accuracy = scores.zip(yb.toList()) { score, yi ->
            if ((yi > 0) == (score.data > 0)) 1 else 0
        }.sum().toDouble() / scores.size

        return Pair(totalLoss, accuracy)
    }
}