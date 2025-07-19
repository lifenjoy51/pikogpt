package grad

import Value
import kotlin.random.Random

class Neuron(
    nin: Int,
    private val nonlin: Boolean = true
) {

    val w: List<Value> = List(nin) { Value(Random.nextDouble(-1.0, 1.0).toFloat()) }
    val b: Value = Value(0.0f)

    operator fun invoke(x: List<Value>): Value {
        var act = b
        for ((wi, xi) in w.zip(x)) {
            act = act + wi * xi
        }
        return if (nonlin) act.relu() else act
    }

    fun parameters(): List<Value> {
        return w + b
    }

    override fun toString(): String {
        return "${if (nonlin) "ReLU" else "gpt.Linear"}grad.Neuron(${w.size})"
    }
}