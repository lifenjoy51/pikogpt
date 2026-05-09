import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * 그래디언트 기록 컨텍스트.
 *
 * PyTorch의 `torch.no_grad()`와 동일한 개념입니다. `enabled=true`일 때 `Value` 연산은
 * 계산 그래프(부모 참조 + backward 클로저)를 구축합니다. `GradContext.noGrad { ... }`
 * 블록 안에서는 `enabled=false`가 되어 그래프 구축을 건너뛰고 스칼라 값만 계산합니다.
 *
 * 평가/추론 경로에서 불필요한 그래프 할당과 GC 압력을 제거하기 위해 사용합니다.
 *
 * 스레드별 상태는 `ThreadLocal`로 분리되어 있어 `Dispatchers.Default` 같은 멀티 스레드
 * 디스패처에서 여러 코루틴이 동시에 사용해도 서로의 플래그를 침범하지 않습니다.
 * 다만 `noGrad` 블록 내부에서 `suspend` 함수를 호출해 스레드가 바뀌는 경우까지 보호하지는
 * 않으므로, 블록 안에서는 non-suspending 동기 계산만 수행한다고 가정합니다.
 */
object GradContext {
    private val enabledPerThread = ThreadLocal.withInitial { true }

    /** 현재 스레드에서 그래디언트 추적이 활성화되어 있는지. */
    val enabled: Boolean get() = enabledPerThread.get()

    /**
     * 주어진 블록을 그래디언트 추적이 꺼진 상태로 실행합니다.
     * 블록이 예외로 종료되어도 이전 상태로 복원됩니다.
     */
    fun <T> noGrad(block: () -> T): T {
        val previous = enabledPerThread.get()
        enabledPerThread.set(false)
        try {
            return block()
        } finally {
            enabledPerThread.set(previous)
        }
    }
}

/**
 * 자동 미분을 지원하는 스칼라 값.
 *
 * micrograd 직계 포팅. forward 시 `_parentNodes` + `backwardFunction`로 동적 계산 그래프를
 * 구축하고, `backward()`가 위상정렬 후 chain rule로 모든 노드에 그래디언트를 누적합니다.
 *
 * 각 연산자 메서드의 docstring은 다음 두 줄을 표준 헤더로 둡니다:
 *   forward:  out = f(a, b)
 *   backward: ∂out/∂a = ..., ∂out/∂b = ...
 */
class Value(
    var scalarValue: Float,
    private var _parentNodes: Set<Value>? = null
) {
    /** 이 노드에 대한 그래디언트 (역전파로 누적). */
    var gradient: Float = 0.0f

    /** 역전파 시 부모 노드의 gradient에 chain rule 항을 더하는 클로저. */
    var backwardFunction: () -> Unit = {}

    private val parentNodes: Set<Value> get() = _parentNodes ?: emptySet()

    companion object {
        /** 자주 쓰이는 상수. attention 마스킹/초기값 등에서 직접 참조. */
        val ZERO = Value(0.0f)
        val ONE = Value(1.0f)
        val MINUS_ONE = Value(-1.0f)
        val HALF = Value(0.5f)
        val MIN = Value(-1e9f)

        /** GELU에서 매 호출마다 새 Value를 만들지 않도록 hoist된 상수. */
        val GELU_SQRT_2_PI: Value = Value(sqrt(2.0 / PI).toFloat())
        val GELU_C: Value = Value(0.044715f)
    }

    // =================================
    // 산술 연산자
    // =================================

    /**
     * forward:  out = a + b
     * backward: ∂out/∂a = 1, ∂out/∂b = 1
     */
    operator fun plus(rightOperand: Value): Value {
        val resultValue = Value(this.scalarValue + rightOperand.scalarValue)
        if (GradContext.enabled) {
            resultValue._parentNodes = setOf(this, rightOperand)
            resultValue.backwardFunction = {
                this.gradient += resultValue.gradient
                rightOperand.gradient += resultValue.gradient
            }
        }
        return resultValue
    }

    operator fun plus(number: Number): Value = this + Value(number.toFloat())

    /**
     * forward:  out = a * b
     * backward: ∂out/∂a = b, ∂out/∂b = a
     */
    operator fun times(rightOperand: Value): Value {
        val resultValue = Value(this.scalarValue * rightOperand.scalarValue)
        if (GradContext.enabled) {
            resultValue._parentNodes = setOf(this, rightOperand)
            resultValue.backwardFunction = {
                this.gradient += rightOperand.scalarValue * resultValue.gradient
                rightOperand.gradient += this.scalarValue * resultValue.gradient
            }
        }
        return resultValue
    }

    operator fun times(number: Number): Value = this * Value(number.toFloat())

    /**
     * forward:  out = -a
     * backward: ∂out/∂a = -1
     *
     * `a * (-1)`로 곱셈의 backward에 위임.
     */
    operator fun unaryMinus(): Value = this * MINUS_ONE

    /**
     * forward:  out = a - b
     * backward: a + (-b)의 backward로 위임 (덧셈 + 단항 마이너스)
     */
    operator fun minus(rightOperand: Value): Value = this + -rightOperand

    operator fun minus(number: Number): Value = this - Value(number.toFloat())

    /**
     * forward:  out = a / b
     * backward: ∂out/∂a = 1/b, ∂out/∂b = -a / b²
     */
    operator fun div(denominator: Value): Value {
        val resultValue = Value(this.scalarValue / denominator.scalarValue)
        if (GradContext.enabled) {
            resultValue._parentNodes = setOf(this, denominator)
            resultValue.backwardFunction = {
                this.gradient += (1.0f / denominator.scalarValue) * resultValue.gradient
                denominator.gradient += (-this.scalarValue / (denominator.scalarValue * denominator.scalarValue)) * resultValue.gradient
            }
        }
        return resultValue
    }

    operator fun div(number: Number): Value = this / Value(number.toFloat())

    /**
     * forward:  out = a^n
     * backward: ∂out/∂a = n * a^(n-1)
     */
    fun pow(exponent: Float): Value {
        val resultValue = Value(this.scalarValue.pow(exponent))
        if (GradContext.enabled) {
            resultValue._parentNodes = setOf(this)
            resultValue.backwardFunction = {
                this.gradient += (exponent * this.scalarValue.pow(exponent - 1)) * resultValue.gradient
            }
        }
        return resultValue
    }

    // =================================
    // 활성화 함수
    // =================================

    /**
     * forward:  out = max(0, a)
     * backward: ∂out/∂a = 1 if a > 0 else 0
     */
    fun relu(): Value {
        val activatedValue = Value(if (this.scalarValue < 0) 0.0f else this.scalarValue)
        if (GradContext.enabled) {
            activatedValue._parentNodes = setOf(this)
            activatedValue.backwardFunction = {
                this.gradient += (if (activatedValue.scalarValue > 0) 1.0f else 0.0f) * activatedValue.gradient
            }
        }
        return activatedValue
    }

    /**
     * forward:  out = exp(a)
     * backward: ∂out/∂a = exp(a) = out
     */
    fun exp(): Value {
        val exponentialResult = Value(exp(this.scalarValue.toDouble()).toFloat())
        if (GradContext.enabled) {
            exponentialResult._parentNodes = setOf(this)
            exponentialResult.backwardFunction = {
                this.gradient += exponentialResult.scalarValue * exponentialResult.gradient
            }
        }
        return exponentialResult
    }

    /**
     * forward:  out = 1 / (1 + exp(-a))
     * backward: chain rule로 자동 (div + exp + neg + plus 조합).
     *
     * 별도 backward 클로저를 두지 않고 합성된 sub-그래프가 그대로 미분되도록 둡니다.
     */
    fun sigmoid(): Value = ONE / (ONE + (-this).exp())

    /**
     * forward:  out = 0.5 * a * (1 + tanh(sqrt(2/π) * (a + 0.044715 * a³)))
     * backward: chain rule로 자동 (모든 sub-연산이 Value 그래프).
     *
     * 상수 (sqrt(2/π), 0.044715)는 companion에 hoist되어 매 호출마다 새 노드를 만들지 않습니다.
     * 그러나 결과 식 자체는 sub-Value 그래프이므로 chain rule이 그대로 작동합니다.
     */
    fun gelu(): Value {
        val inputValue = this

        // 내부 식: sqrt(2/π) * (a + 0.044715 * a³)
        val innerExpression = GELU_SQRT_2_PI * (inputValue + GELU_C * inputValue.pow(3.0f))

        // tanh(y) = (exp(2y) - 1) / (exp(2y) + 1)
        val exp2y = (innerExpression * 2.0f).exp()
        val tanhValue = (exp2y - ONE) / (exp2y + ONE)

        return inputValue * HALF * (ONE + tanhValue)
    }

    // =================================
    // 역전파
    // =================================

    /**
     * 역전파.
     *
     * 1. 계산 그래프를 DFS로 위상정렬 (parents → child 순)
     * 2. 출력 노드 gradient를 1로 초기화
     * 3. 역순으로 각 노드의 backwardFunction 실행 → chain rule로 부모 gradient에 누적
     *
     * @param clearGraph true면 backward 종료 후 부모 참조와 클로저를 해제하여 GC 부담을 줄입니다.
     *                   default false (micrograd 원본과 같이 그래프 보존). 학습 루프에서 메모리
     *                   압박이 있을 때만 명시적으로 true를 넘깁니다.
     */
    fun backward(clearGraph: Boolean = false) {
        val topologicalOrder = mutableListOf<Value>()
        val visitedNodes = mutableSetOf<Value>()

        fun buildTopologicalOrder(currentNode: Value) {
            if (currentNode !in visitedNodes) {
                visitedNodes.add(currentNode)
                currentNode.parentNodes.forEach(::buildTopologicalOrder)
                topologicalOrder.add(currentNode)
            }
        }

        buildTopologicalOrder(this)

        this.gradient = 1.0f
        topologicalOrder.reversed().forEach { node -> node.backwardFunction() }

        if (clearGraph) clearComputationGraph(topologicalOrder)
    }

    private fun clearComputationGraph(nodes: List<Value>) {
        nodes.forEach { node ->
            node._parentNodes = null
            node.backwardFunction = {}
        }
    }

    override fun toString(): String = "Value(scalarValue=$scalarValue, gradient=$gradient)"

    // =================================
    // Number ↔ Value 자연스러운 문법용 확장
    // =================================

    operator fun Number.plus(valueObject: Value): Value = valueObject + this
    operator fun Number.times(valueObject: Value): Value = valueObject * this
    operator fun Number.minus(valueObject: Value): Value = Value(this.toFloat()) - valueObject
    operator fun Number.div(valueObject: Value): Value = Value(this.toFloat()) / valueObject
}
