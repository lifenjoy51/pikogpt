package grad

import Value

class Layer(
    nin: Int,
    nout: Int,
    nonlin: Boolean = true
) : Module() {

    val neurons: List<Neuron> = List(nout) { Neuron(nin, nonlin) }

    operator fun invoke(x: List<Value>): Any {
        val out = neurons.map { n -> n(x) }
        return if (out.size == 1) out[0] else out
    }

    override fun parameters(): List<Value> {
        return neurons.flatMap { it.parameters() }
    }

    override fun toString(): String {
        return "grad.Layer of [${neurons.joinToString(", ")}]"
    }
}