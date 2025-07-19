package grad

import Value
import kotlin.random.Random

/**
 * 인공 뉴런 (Artificial Neuron)
 *
 * 신경망의 기본 구성 단위입니다.
 * 여러 개의 입력(input)을 받아 각각에 가중치(weight)를 곱하고, 편향(bias)을 더한 후,
 * (선택적으로) 비선형 활성화 함수(ReLU)를 통과시켜 출력을 생성합니다.
 *
 * @param numberOfInputs 뉴런에 들어오는 입력의 수
 * @param applyNonlinearity 비선형 활성화 함수(ReLU)를 적용할지 여부 (기본값: true)
 */
class Neuron(
    numberOfInputs: Int,
    private val applyNonlinearity: Boolean = true
) {
    /** 입력에 곱해지는 가중치들. -1과 1 사이의 무작위 값으로 초기화됩니다. */
    val weights: List<Value> = List(numberOfInputs) { Value(Random.nextDouble(-1.0, 1.0).toFloat()) }

    /** 가중합에 더해지는 편향. 0으로 초기화됩니다. */
    val bias: Value = Value(0.0f)

    /**
     * 뉴런의 순전파를 수행합니다.
     *
     * 입력 값들과 가중치를 곱한 총합에 편향을 더하고, 활성화 함수를 적용합니다.
     * `invoke` 연산자를 오버로딩하여 `neuron(inputs)` 형태로 호출할 수 있습니다.
     *
     * @param inputValues 입력 값들의 리스트 (Value 객체)
     * @return 뉴런의 출력 값 (Value 객체)
     */
    operator fun invoke(inputValues: List<Value>): Value {
        // 가중합 계산: Σ(weight_i * input_i) + bias
        var activation = bias
        for ((weight, inputValue) in weights.zip(inputValues)) {
            activation += weight * inputValue
        }
        // 비선형 활성화 함수 적용 (선택적)
        return if (applyNonlinearity) activation.relu() else activation
    }

    /**
     * 뉴런의 모든 학습 가능한 파라미터를 수집합니다.
     *
     * 가중치들과 편향을 하나의 리스트로 반환합니다.
     *
     * @return 모든 파라미터(Value 객체)의 리스트
     */
    fun parameters(): List<Value> {
        return weights + bias
    }

    /**
     * 뉴런의 정보를 문자열로 표현합니다.
     *
     * 디버깅 및 로깅에 유용합니다.
     *
     * @return 뉴런 정보 문자열 (예: "ReLU grad.Neuron(16)")
     */
    override fun toString(): String {
        return "${if (applyNonlinearity) "ReLU" else "Linear"} Neuron(${weights.size})"
    }
}