package gpt

import Value
import kotlin.random.Random

/**
 * Dropout 레이어 구현
 * 훈련 시에만 활성화되며, 추론 시에는 무시됩니다.
 */
class Dropout(
    private val probability: Float
) {
    companion object {
        // 전역 훈련 모드 플래그
        var training: Boolean = true
    }

    /**
     * Forward pass
     * @param input 입력 Value들의 리스트
     * @return Dropout이 적용된 Value들의 리스트
     */
    fun forward(input: List<Value>): List<Value> {
        if (!training || probability <= 0.0f) {
            return input
        }

        val scale = 1.0f / (1.0f - probability)
        
        return input.map { value ->
            if (Random.nextFloat() < probability) {
                Value(0.0f)
            } else {
                value * Value(scale)
            }
        }
    }

    /**
     * 2D 입력에 대한 forward pass
     * @param input 2D Value 배열
     * @return Dropout이 적용된 2D Value 배열
     */
    fun forward(input: Array<Array<Value>>): Array<Array<Value>> {
        if (!training || probability <= 0.0f) {
            return input
        }

        val scale = 1.0f / (1.0f - probability)
        
        return Array(input.size) { i ->
            Array(input[i].size) { j ->
                if (Random.nextFloat() < probability) {
                    Value(0.0f)
                } else {
                    input[i][j] * Value(scale)
                }
            }
        }
    }
}