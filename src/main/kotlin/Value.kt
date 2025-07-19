import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * 자동 미분을 지원하는 스칼라 값 클래스
 *
 * 이 클래스는 미니 그래드 엔진의 핵심으로, 모든 수치 계산에 대해 자동으로 그래디언트를 추적합니다.
 * 순전파 계산 중에 계산 그래프를 구축하고, 역전파 시 연쇄 법칙(chain rule)을 적용하여 그래디언트를 계산합니다.
 *
 * @param scalarValue 이 노드가 저장하는 실제 스칼라 값
 * @param parentNodes 이 값을 생성한 부모 노드들의 집합 (계산 그래프 구성용)
 */
class Value(
    var scalarValue: Float,
    private val parentNodes: Set<Value> = emptySet()
) {
    /** 이 노드에 대한 그래디언트 값 (역전파로 계산됨) */
    var gradient: Float = 0.0f

    /** 역전파 시 실행될 함수 (각 연산마다 고유한 그래디언트 계산 로직) */
    var backwardFunction: () -> Unit = {}

    // --- 연산자 오버로딩 최적화 ---

    /**
     * 두 Value 객체의 덧셈 연산
     *
     * out = leftOperand + rightOperand
     * 역전파: ∂out/∂leftOperand = 1, ∂out/∂rightOperand = 1
     *
     * @param rightOperand 덧셈의 오른쪽 피연산자
     * @return 덧셈 결과를 담은 새로운 Value 객체
     */
    operator fun plus(rightOperand: Value): Value {
        val resultValue = Value(this.scalarValue + rightOperand.scalarValue, setOf(this, rightOperand))
        resultValue.backwardFunction = {
            // 덧셈의 로컬 그래디언트는 항상 1이므로, 출력 그래디언트를 그대로 전파
            this.gradient += resultValue.gradient
            rightOperand.gradient += resultValue.gradient
        }
        return resultValue
    }

    /**
     * Value와 숫자의 덧셈 연산
     *
     * @param number 더할 숫자 값
     * @return 덧셈 결과를 담은 새로운 Value 객체
     */
    operator fun plus(number: Number): Value = this + Value(number.toFloat())

    /**
     * 두 Value 객체의 곱셈 연산
     *
     * out = leftOperand * rightOperand
     * 역전파: ∂out/∂leftOperand = rightOperand, ∂out/∂rightOperand = leftOperand
     *
     * @param rightOperand 곱셈의 오른쪽 피연산자
     * @return 곱셈 결과를 담은 새로운 Value 객체
     */
    operator fun times(rightOperand: Value): Value {
        val resultValue = Value(this.scalarValue * rightOperand.scalarValue, setOf(this, rightOperand))
        resultValue.backwardFunction = {
            // 곱셈의 로컬 그래디언트: 각 피연산자에 대해 상대방의 값
            this.gradient += rightOperand.scalarValue * resultValue.gradient
            rightOperand.gradient += this.scalarValue * resultValue.gradient
        }
        return resultValue
    }

    /**
     * Value와 숫자의 곱셈 연산
     *
     * @param number 곱할 숫자 값
     * @return 곱셈 결과를 담은 새로운 Value 객체
     */
    operator fun times(number: Number): Value = this * Value(number.toFloat())

    /**
     * 단항 마이너스 연산 (부호 반전)
     *
     * @return 부호가 반전된 새로운 Value 객체
     */
    operator fun unaryMinus(): Value = this * -1.0f

    /**
     * 두 Value 객체의 뺄셈 연산
     *
     * 덧셈과 단항 마이너스의 조합으로 구현됨
     *
     * @param rightOperand 뺄셈의 오른쪽 피연산자
     * @return 뺄셈 결과를 담은 새로운 Value 객체
     */
    operator fun minus(rightOperand: Value): Value = this + -rightOperand

    /**
     * Value와 숫자의 뺄셈 연산
     *
     * @param number 뺄 숫자 값
     * @return 뺄셈 결과를 담은 새로운 Value 객체
     */
    operator fun minus(number: Number): Value = this - Value(number.toFloat())

    /**
     * 두 Value 객체의 나눗셈 연산
     *
     * out = numerator / denominator
     * 역전파: ∂out/∂numerator = 1/denominator, ∂out/∂denominator = -numerator/(denominator²)
     *
     * @param denominator 나눗셈의 분모
     * @return 나눗셈 결과를 담은 새로운 Value 객체
     */
    operator fun div(denominator: Value): Value {
        val resultValue = Value(this.scalarValue / denominator.scalarValue, setOf(this, denominator))
        resultValue.backwardFunction = {
            // 나눗셈의 로컬 그래디언트
            this.gradient += (1.0f / denominator.scalarValue) * resultValue.gradient
            denominator.gradient += (-this.scalarValue / (denominator.scalarValue * denominator.scalarValue)) * resultValue.gradient
        }
        return resultValue
    }

    /**
     * Value를 숫자로 나누는 연산
     *
     * @param number 나눌 숫자 값
     * @return 나눗셈 결과를 담은 새로운 Value 객체
     */
    operator fun div(number: Number): Value = this / Value(number.toFloat())

    /**
     * 거듭제곱 연산
     *
     * out = base^exponent
     * 역전파: ∂out/∂base = exponent * base^(exponent-1)
     *
     * @param exponent 지수 값
     * @return 거듭제곱 결과를 담은 새로운 Value 객체
     */
    fun pow(exponent: Float): Value {
        val resultValue = Value(this.scalarValue.pow(exponent), setOf(this))
        resultValue.backwardFunction = {
            // 거듭제곱의 로컬 그래디언트: exponent * base^(exponent-1)
            this.gradient += (exponent * this.scalarValue.pow(exponent - 1)) * resultValue.gradient
        }
        return resultValue
    }

    // =================================
    // 활성화 함수들 (Activation Functions)
    // =================================

    /**
     * ReLU (Rectified Linear Unit) 활성화 함수
     *
     * ReLU(x) = max(0, x)
     * 역전파: ∂ReLU/∂x = 1 if x > 0, else 0
     *
     * 음수 값을 0으로 설정하여 비선형성을 도입하고 그래디언트 소실 문제를 완화합니다.
     *
     * @return ReLU 적용 결과를 담은 새로운 Value 객체
     */
    fun relu(): Value {
        val activatedValue = Value(if (this.scalarValue < 0) 0.0f else this.scalarValue, setOf(this))
        activatedValue.backwardFunction = {
            // ReLU의 로컬 그래디언트: 입력이 양수면 1, 음수면 0
            this.gradient += (if (activatedValue.scalarValue > 0) 1.0f else 0.0f) * activatedValue.gradient
        }
        return activatedValue
    }

    /**
     * 지수 함수 (Exponential function)
     *
     * exp(x) = e^x
     * 역전파: ∂exp/∂x = exp(x)
     *
     * 소프트맥스 계산이나 시그모이드 활성화 함수에서 사용됩니다.
     *
     * @return 지수 함수 결과를 담은 새로운 Value 객체
     */
    fun exp(): Value {
        val exponentialResult = Value(exp(this.scalarValue.toDouble()).toFloat(), setOf(this))
        exponentialResult.backwardFunction = {
            // 지수 함수의 로컬 그래디언트: exp(x)
            this.gradient += exponentialResult.scalarValue * exponentialResult.gradient
        }
        return exponentialResult
    }

    /**
     * 시그모이드 활성화 함수
     *
     * sigmoid(x) = 1 / (1 + exp(-x))
     * 역전파: ∂sigmoid/∂x = sigmoid(x) * (1 - sigmoid(x))
     *
     * 0과 1 사이의 값으로 매핑하여 확률이나 게이트 역할에 사용됩니다.
     *
     * @return 시그모이드 결과를 담은 새로운 Value 객체
     */
    fun sigmoid(): Value {
        // sigmoid(x) = 1 / (1 + exp(-x))
        // 체인 룰(chain rule)에 의해 자동으로 역전파 처리됨
        val sigmoidResult = 1.0f / (1.0f + (-this).exp())
        return sigmoidResult
    }

    /**
     * GELU (Gaussian Error Linear Unit) 활성화 함수
     *
     * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
     *
     * 가우시안 오차 함수를 기반으로 한 활성화 함수로, ReLU보다 부드러운 전환을 제공합니다.
     * Transformer 모델에서 널리 사용되는 활성화 함수입니다.
     *
     * @return GELU 결과를 담은 새로운 Value 객체
     */
    fun gelu(): Value {
        val inputValue = this
        val sqrt2OverPi = Value(sqrt(2.0 / PI).toFloat())  // sqrt(2/π)
        val geluConstant = Value(0.044715f)                // GELU 상수

        // GELU 내부 식: sqrt(2/π) * (x + 0.044715 * x³)
        val innerExpression = sqrt2OverPi * (inputValue + geluConstant * inputValue.pow(3.0f))

        // tanh(y) = (exp(2y) - 1) / (exp(2y) + 1) 공식을 사용하여 tanh 구현
        val exp2y = (innerExpression * 2.0f).exp()
        val tanhValue = (exp2y - 1.0f) / (exp2y + 1.0f)

        // GELU 최종 결과: 0.5 * x * (1 + tanh(...))
        return inputValue * 0.5f * (1.0f + tanhValue)
    }


    // =================================
    // 역전파 (Backpropagation)
    // =================================

    /**
     * 역전파 알고리즘 실행
     *
     * 계산 그래프를 위상정렬(topological sort)하여 올바른 순서로 그래디언트를 전파합니다.
     * 연쇄 법칙(chain rule)을 사용하여 모든 리프 노드의 그래디언트를 계산합니다.
     *
     * 알고리즘 단계:
     * 1. DFS로 계산 그래프를 위상정렬
     * 2. 출력 노드의 그래디언트를 1.0으로 초기화
     * 3. 역순으로 각 노드의 그래디언트 전파 함수 실행
     */
    fun backward() {
        val topologicalOrder = mutableListOf<Value>()
        val visitedNodes = mutableSetOf<Value>()

        /**
         * 계산 그래프를 DFS로 순회하여 위상정렬 수행
         *
         * @param currentNode 현재 방문 중인 노드
         */
        fun buildTopologicalOrder(currentNode: Value) {
            if (currentNode !in visitedNodes) {
                visitedNodes.add(currentNode)
                // 모든 부모 노드를 재귀적으로 방문
                currentNode.parentNodes.forEach(::buildTopologicalOrder)
                // 마지막에 현재 노드를 추가 (위상정렬)
                topologicalOrder.add(currentNode)
            }
        }

        buildTopologicalOrder(this)

        // 출력 노드(최종 노드)의 그래디언트를 1.0으로 설정
        this.gradient = 1.0f

        // 역순으로 각 노드의 그래디언트 전파 함수 실행
        topologicalOrder.reversed().forEach { node -> node.backwardFunction() }
    }

    /**
     * 디버깅을 위한 문자열 표현
     *
     * Value 객체의 현재 상태를 사람이 읽기 쉽은 형태로 변환합니다.
     * 스칼라 값과 그래디언트 값을 동시에 표시하여 디버깅에 도움을 줍니다.
     *
     * @return 데이터와 그래디언트 값을 포함한 문자열
     */
    override fun toString(): String = "Value(scalarValue=$scalarValue, gradient=$gradient)"

    // =================================
    // 확장 함수 - 자연스러운 연산을 위한 유틸리티
    // =================================

    /**
     * 숫자 + Value 연산을 위한 확장 함수
     * 3.0 + valueObject 같은 자연스러운 문법을 지원합니다.
     */
    operator fun Number.plus(valueObject: Value): Value = valueObject + this

    /**
     * 숫자 * Value 연산을 위한 확장 함수
     * 2.0 * valueObject 같은 자연스러운 문법을 지원합니다.
     */
    operator fun Number.times(valueObject: Value): Value = valueObject * this

    /**
     * 숫자 - Value 연산을 위한 확장 함수
     * 5.0 - valueObject 같은 자연스러운 문법을 지원합니다.
     */
    operator fun Number.minus(valueObject: Value): Value = Value(this.toFloat()) - valueObject

    /**
     * 숫자 / Value 연산을 위한 확장 함수
     * 10.0 / valueObject 같은 자연스러운 문법을 지원합니다.
     */
    operator fun Number.div(valueObject: Value): Value = Value(this.toFloat()) / valueObject
}
