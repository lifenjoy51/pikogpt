package grad

import Value

class MLP(
    numberOfInputs: Int,
    layerOutputSizes: List<Int>
) {

    private val layerSizes: List<Int> = listOf(numberOfInputs) + layerOutputSizes
    val layers: List<Layer> = List(layerOutputSizes.size) { layerIndex ->
        Layer(layerSizes[layerIndex], layerSizes[layerIndex + 1], nonlinear = layerIndex != layerOutputSizes.size - 1)
    }

    fun zeroGrad() {
        for (parameter in parameters()) {
            parameter.grad = 0.0f
        }
    }

    operator fun invoke(inputValues: List<Value>): Any {
        var currentOutput: Any = inputValues
        for (layer in layers) {
            currentOutput = when (currentOutput) {
                is List<*> -> {
                    @Suppress("UNCHECKED_CAST")
                    layer(currentOutput as List<Value>)
                }
                is Value -> layer(listOf(currentOutput))
                else -> throw IllegalArgumentException("Unexpected type")
            }
        }
        return currentOutput
    }

    fun parameters(): List<Value> {
        return layers.flatMap { it.parameters() }
    }

    override fun toString(): String {
        return "grad.MLP of [${layers.joinToString(", ")}]"
    }
}