package grad

import Value

/**
 * 손실 및 정확도 계산기
 *
 * 주어진 모델과 데이터에 대해 손실(loss)과 정확도(accuracy)를 계산하는 역할을 합니다.
 * SVM(Support Vector Machine)의 "max-margin" 손실 함수와 L2 정규화를 함께 사용합니다.
 *
 * @param model 훈련 및 평가에 사용될 MLP 모델
 * @param trainingFeatures 훈련 데이터의 특징(feature) 배열
 * @param trainingLabels 훈련 데이터의 레이블(label) 배열
 */
class LossCalculator(
    private val model: MLP,
    private val trainingFeatures: Array<FloatArray>,
    private val trainingLabels: IntArray
) {
    /**
     * 모델의 손실과 정확도를 계산합니다.
     *
     * 1. (선택적) 배치 샘플링: `batchSize`가 주어지면 전체 데이터에서 무작위로 배치를 선택합니다.
     * 2. 순전파: 모델을 통해 예측 점수(score)를 계산합니다.
     * 3. 데이터 손실 계산: SVM "max-margin" 손실을 사용하여 각 데이터 포인트의 손실을 계산합니다.
     *    - 손실 = max(0, 1 - y_i * f(x_i)), 여기서 y_i는 실제 레이블, f(x_i)는 예측 점수입니다.
     * 4. 정규화 손실 계산: 모델 파라미터에 L2 정규화를 적용하여 과적합을 방지합니다.
     * 5. 총 손실: 데이터 손실과 정규화 손실을 합산합니다.
     * 6. 정확도 계산: 예측 점수의 부호와 실제 레이블의 부호가 일치하는 비율을 계산합니다.
     *
     * @param batchSize 계산에 사용할 배치의 크기. null이면 전체 데이터를 사용합니다.
     * @return Pair(총 손실(Value), 정확도(Double))
     */
    fun loss(batchSize: Int? = null): Pair<Value, Double> {
        // 1. 배치 선택 (batchSize가 null이면 전체 데이터 사용)
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

        // 2. 모델 순전파
        val modelScores = valueInputs.map { inputVector ->
            model(inputVector) as Value
        }

        // 3. SVM "max-margin" 데이터 손실 계산
        val individualLosses = modelScores.zip(batchLabels.toList()) { predictedScore, actualLabel ->
            // loss = max(0, 1 - label * score)
            (Value(1.0f) + Value(-actualLabel.toFloat()) * predictedScore).relu()
        }
        val averageDataLoss = individualLosses.reduce { acc, loss -> acc + loss } * Value(1.0f / individualLosses.size)

        // 4. L2 정규화 손실 계산
        val regularizationWeight = 1e-4f
        val regularizationLoss = model.parameters()
            .map { it * it } // 제곱
            .reduce { acc, sqParam -> acc + sqParam } * Value(regularizationWeight)

        // 5. 총 손실 = 데이터 손실 + 정규화 손실
        val totalLoss = averageDataLoss + regularizationLoss

        // 6. 정확도 계산
        val accuracy = modelScores.zip(batchLabels.toList()) { predictedScore, actualLabel ->
            if ((actualLabel > 0) == (predictedScore.scalarValue > 0)) 1 else 0
        }.sum().toDouble() / modelScores.size

        return Pair(totalLoss, accuracy)
    }
}