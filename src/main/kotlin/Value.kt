import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.math.PI
import kotlin.math.*
import kotlin.random.Random

class Value(
    var data: Double,
    private val _children: Set<Value> = emptySet(),
    var _op: String = ""
) {
    var grad: Double = 0.0
    var _backward: () -> Unit = {}

    // --- 연산자 오버로딩 최적화 ---

    operator fun plus(other: Value): Value {
        val out = Value(this.data + other.data, setOf(this, other), "+")
        out._backward = {
            this.grad += out.grad
            other.grad += out.grad
        }
        return out
    }

    operator fun plus(other: Number): Value = this + Value(other.toDouble())

    operator fun times(other: Value): Value {
        val out = Value(this.data * other.data, setOf(this, other), "*")
        out._backward = {
            this.grad += other.data * out.grad
            other.grad += this.data * out.grad
        }
        return out
    }

    operator fun times(other: Number): Value = this * Value(other.toDouble())

    operator fun unaryMinus(): Value = this * -1.0

    operator fun minus(other: Value): Value = this + -other

    operator fun minus(other: Number): Value = this - Value(other.toDouble())

    // pow(-1) 대신 직접 나누기 구현
    operator fun div(other: Value): Value {
        val out = Value(this.data / other.data, setOf(this, other), "/")
        out._backward = {
            this.grad += (1.0 / other.data) * out.grad
            other.grad += (-this.data / (other.data * other.data)) * out.grad
        }
        return out
    }

    operator fun div(other: Number): Value = this / Value(other.toDouble())

    fun pow(other: Double): Value {
        val out = Value(this.data.pow(other), setOf(this), "**$other")
        out._backward = {
            this.grad += (other * this.data.pow(other - 1)) * out.grad
        }
        return out
    }

    // --- 활성화 함수 (Utils에서 이동) ---

    fun relu(): Value {
        val out = Value(if (this.data < 0) 0.0 else this.data, setOf(this), "ReLU")
        out._backward = {
            this.grad += (if (out.data > 0) 1.0 else 0.0) * out.grad
        }
        return out
    }

    fun exp(): Value {
        val out = Value(exp(this.data), setOf(this), "exp")
        out._backward = {
            this.grad += out.data * out.grad
        }
        return out
    }

    fun sigmoid(): Value {
        // sigmoid(x) = 1 / (1 + exp(-x))
        val out = 1.0 / (1.0 + (-this).exp())
        // _backward는 체인룰에 의해 자동으로 처리됨
        return out
    }

    fun gelu(): Value {
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        val x = this
        val c1 = Value(sqrt(2.0 / PI))
        val c2 = Value(0.044715)

        val inner = c1 * (x + c2 * x.pow(3.0))
        // Kotlin의 tanh는 Value 객체에 대해 정의되어 있지 않으므로, tanh를 직접 구현하거나 근사해야 합니다.
        // 여기서는 tanh(y) = (exp(2y) - 1) / (exp(2y) + 1) 공식을 사용합니다.
        val exp2y = (inner * 2.0).exp()
        val tanhInner = (exp2y - 1.0) / (exp2y + 1.0)

        return x * 0.5 * (1.0 + tanhInner)
    }


    // --- 역전파 ---

    fun backward() {
        val topo = mutableListOf<Value>()
        val visited = mutableSetOf<Value>()
        fun buildTopo(v: Value) {
            if (v !in visited) {
                visited.add(v)
                v._children.forEach(::buildTopo)
                topo.add(v)
            }
        }
        buildTopo(this)

        this.grad = 1.0
        topo.reversed().forEach { it._backward() }
    }

    override fun toString(): String = "Value(data=$data, grad=$grad)"

    // 연산을 더 자연스럽게 하기 위한 확장 함수
    operator fun Number.plus(other: Value): Value = other + this
    operator fun Number.times(other: Value): Value = other * this
    operator fun Number.minus(other: Value): Value = other - this
    operator fun Number.div(other: Value): Value = other / this
}
