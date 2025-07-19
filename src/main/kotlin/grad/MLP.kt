package grad

import Value

/**
 * 다층 퍼셉트론 (Multi-Layer Perceptron)
 *
 * 여러 개의 완전 연결 레이어(Layer)로 구성된 신경망입니다.
 * 입력부터 출력까지 순차적으로 데이터를 처리하며, 각 레이어는 비선형 활성화 함수를 포함할 수 있습니다.
 *
 * @param numberOfInputs 입력 데이터의 차원 수
 * @param layerOutputSizes 각 은닉층과 출력층의 뉴런(출력) 수 리스트
 */
class MLP(
    numberOfInputs: Int,
    layerOutputSizes: List<Int>
) {
    /** 각 레이어의 크기 (입력층 포함) */
    private val layerSizes: List<Int> = listOf(numberOfInputs) + layerOutputSizes

    /** MLP를 구성하는 레이어들의 리스트 */
    val layers: List<Layer> = List(layerOutputSizes.size) { layerIndex ->
        Layer(
            inputSize = layerSizes[layerIndex],
            outputSize = layerSizes[layerIndex + 1],
            // 마지막 레이어를 제외하고는 비선형 활성화 함수(ReLU) 적용
            applyNonlinearity = layerIndex != layerOutputSizes.size - 1
        )
    }

    /**
     * 모든 파라미터의 그래디언트를 0으로 초기화합니다.
     *
     * 역전파를 수행하기 전에 이전 그래디언트 값을 제거하기 위해 호출해야 합니다.
     */
    fun zeroGrad() {
        for (parameter in parameters()) {
            parameter.gradient = 0.0f
        }
    }

    /**
     * MLP의 순전파를 수행합니다.
     *
     * 입력 데이터를 받아 모든 레이어를 순차적으로 통과시킨 후 최종 출력을 반환합니다.
     * `invoke` 연산자를 오버로딩하여 `mlp(input)` 형태로 호출할 수 있습니다.
     *
     * @param inputValues 입력 데이터 (Value 객체의 리스트)
     * @return 최종 레이어의 출력 (Value 또는 Value 리스트)
     */
    operator fun invoke(inputValues: List<Value>): Any {
        var currentOutput: Any = inputValues
        for (layer in layers) {
            currentOutput = when (currentOutput) {
                is List<*> -> {
                    @Suppress("UNCHECKED_CAST")
                    layer(currentOutput as List<Value>)
                }
                is Value -> layer(listOf(currentOutput))
                else -> throw IllegalArgumentException("Unexpected type for layer input")
            }
        }
        return currentOutput
    }

    /**
     * MLP의 모든 학습 가능한 파라미터를 수집합니다.
     *
     * 모든 레이어에 포함된 가중치와 편향을 하나의 리스트로 반환합니다.
     * 옵티마이저가 이 파라미터들을 업데이트하는 데 사용됩니다.
     *
     * @return 모든 파라미터(Value 객체)의 리스트
     */
    fun parameters(): List<Value> {
        return layers.flatMap { it.parameters() }
    }

    /**
     * MLP의 구조를 문자열로 표현합니다.
     *
     * 디버깅 및 로깅에 유용합니다.
     *
     * @return MLP 구조 정보 문자열 (예: "grad.MLP of [Layer(in=2, out=16), ...]")
     */
    override fun toString(): String {
        return "grad.MLP of [${layers.joinToString(", ")}]"
    }
}