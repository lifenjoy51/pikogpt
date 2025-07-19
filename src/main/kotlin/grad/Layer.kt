package grad

import Value

/**
 * 뉴런들의 집합으로 구성된 완전 연결 레이어 (Fully Connected Layer)
 *
 * 여러 개의 뉴런(Neuron)을 병렬로 연결하여 입력 벡터를 출력 벡터로 변환합니다.
 * 다층 퍼셉트론(MLP)의 기본 구성 요소입니다.
 *
 * @param inputSize 레이어에 들어오는 입력의 수 (각 뉴런의 입력 수)
 * @param outputSize 레이어의 출력 수 (레이어에 포함된 뉴런의 수)
 * @param applyNonlinearity 각 뉴런에 비선형 활성화 함수(ReLU)를 적용할지 여부
 */
class Layer(
    inputSize: Int,
    outputSize: Int,
    applyNonlinearity: Boolean = true
) {
    /** 이 레이어를 구성하는 뉴런들의 리스트 */
    val neurons: List<Neuron> = List(outputSize) { Neuron(inputSize, applyNonlinearity) }

    /**
     * 레이어의 순전파를 수행합니다.
     *
     * 입력 데이터를 각 뉴런에 전달하고, 모든 뉴런의 출력을 모아 반환합니다.
     * `invoke` 연산자를 오버로딩하여 `layer(inputs)` 형태로 호출할 수 있습니다.
     *
     * @param inputValues 입력 값들의 리스트 (Value 객체)
     * @return 뉴런 출력들의 리스트 (Value 객체). 출력이 하나일 경우 단일 Value 객체를 반환합니다.
     */
    operator fun invoke(inputValues: List<Value>): Any {
        val outputs = neurons.map { it(inputValues) }
        return if (outputs.size == 1) outputs[0] else outputs
    }

    /**
     * 레이어의 모든 학습 가능한 파라미터를 수집합니다.
     *
     * 모든 뉴런에 포함된 가중치와 편향을 하나의 리스트로 반환합니다.
     *
     * @return 모든 파라미터(Value 객체)의 리스트
     */
    fun parameters(): List<Value> {
        return neurons.flatMap { it.parameters() }
    }

    /**
     * 레이어의 정보를 문자열로 표현합니다.
     *
     * 디버깅 및 로깅에 유용합니다.
     *
     * @return 레이어 정보 문자열 (예: "grad.Layer of [ReLU Neuron(2), ...]")
     */
    override fun toString(): String {
        return "grad.Layer of [${neurons.joinToString(", ")}]"
    }
}