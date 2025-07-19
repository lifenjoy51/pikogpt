package grad

import Value

class Layer(
    numberOfInputs: Int,
    numberOfOutputs: Int,
    nonlinear: Boolean = true
) {

    val neurons: List<Neuron> = List(numberOfOutputs) { Neuron(numberOfInputs, nonlinear) }

    operator fun invoke(inputValues: List<Value>): Any {
        val outputs = neurons.map { neuron -> neuron(inputValues) }
        return if (outputs.size == 1) outputs[0] else outputs
    }

    fun parameters(): List<Value> {
        return neurons.flatMap { it.parameters() }
    }

    override fun toString(): String {
        return "grad.Layer of [${neurons.joinToString(", ")}]"
    }
}