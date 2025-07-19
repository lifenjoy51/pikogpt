package grad

import Value
import kotlin.random.Random

class Neuron(
    numberOfInputs: Int,
    private val nonlinear: Boolean = true
) {

    val weights: List<Value> = List(numberOfInputs) { Value(Random.nextDouble(-1.0, 1.0).toFloat()) }
    val bias: Value = Value(0.0f)

    operator fun invoke(inputValues: List<Value>): Value {
        var activation = bias
        for ((weight, inputValue) in weights.zip(inputValues)) {
            activation = activation + weight * inputValue
        }
        return if (nonlinear) activation.relu() else activation
    }

    fun parameters(): List<Value> {
        return weights + bias
    }

    override fun toString(): String {
        return "${if (nonlinear) "ReLU" else "gpt.Linear"}grad.Neuron(${weights.size})"
    }
}