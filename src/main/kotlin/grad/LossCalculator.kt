package grad

import Value

// 손실 함수 구현
class LossCalculator(
    private val model: MLP,
    private val trainingFeatures: Array<FloatArray>,
    private val trainingLabels: IntArray
) {
    fun loss(batchSize: Int? = null): Pair<Value, Double> {
        // 배치 선택
        val (batchFeatures, batchLabels) = if (batchSize == null) {
            Pair(trainingFeatures, trainingLabels)
        } else {
            val randomIndices = (0 until trainingFeatures.size).shuffled().take(batchSize)
            val selectedFeatures = randomIndices.map { trainingFeatures[it] }.toTypedArray()
            val selectedLabels = randomIndices.map { trainingLabels[it] }.toIntArray()
            Pair(selectedFeatures, selectedLabels)
        }

        // 입력 데이터를 Value 객체로 변환
        val valueInputs = batchFeatures.map { featureRow ->
            featureRow.map { Value(it) }
        }

        // 모델 순전파
        val modelScores = valueInputs.map { inputVector ->
            model(inputVector) as Value
        }

        // SVM "max-margin" 손실
        val individualLosses = modelScores.zip(batchLabels.toList()) { predictedScore, actualLabel ->
            (Value(1.0f) + Value(-actualLabel.toFloat()) * predictedScore).relu()
        }

        val averageDataLoss = individualLosses.reduce { accumulator, currentLossValue -> accumulator + currentLossValue } * Value(1.0f / individualLosses.size)

        // L2 정규화
        val regularizationWeight = 1e-4f
        val regularizationLoss = model.parameters()
            .map { parameter -> parameter * parameter }
            .reduce { accumulator, squaredParameter -> accumulator + squaredParameter } * Value(regularizationWeight)

        val totalLoss = averageDataLoss + regularizationLoss

        // 정확도 계산
        val accuracy = modelScores.zip(batchLabels.toList()) { predictedScore, actualLabel ->
            if ((actualLabel > 0) == (predictedScore.data > 0)) 1 else 0
        }.sum().toDouble() / modelScores.size

        return Pair(totalLoss, accuracy)
    }
}