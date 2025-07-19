package grad

import Value

class MLP(
    nin: Int,
    nouts: List<Int>
) {

    private val sz: List<Int> = listOf(nin) + nouts
    val layers: List<Layer> = List(nouts.size) { i ->
        Layer(sz[i], sz[i + 1], nonlin = i != nouts.size - 1)
    }

    fun zeroGrad() {
        for (p in parameters()) {
            p.grad = 0.0f
        }
    }

    operator fun invoke(x: List<Value>): Any {
        var current: Any = x
        for (layer in layers) {
            current = when (current) {
                is List<*> -> {
                    @Suppress("UNCHECKED_CAST")
                    layer(current as List<Value>)
                }
                is Value -> layer(listOf(current))
                else -> throw IllegalArgumentException("Unexpected type")
            }
        }
        return current
    }

    fun parameters(): List<Value> {
        return layers.flatMap { it.parameters() }
    }

    override fun toString(): String {
        return "grad.MLP of [${layers.joinToString(", ")}]"
    }
}