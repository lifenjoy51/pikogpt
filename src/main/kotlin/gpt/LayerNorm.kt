package gpt

import Value

class LayerNorm(
    private val ndim: Int,
    private val bias: Boolean
) {
    private val weight = Array(ndim) { Value(1.0) } // gain
    private val biasParams = if (bias) Array(ndim) { Value(0.0) } else null // bias

    fun forward(input: Array<Value>): Array<Value> {
        val n = input.size.toDouble()
        val mean = input.reduce { acc, v -> acc + v } / n
        val variance = input.map { x -> (x - mean).pow(2.0) }
            .reduce { acc, v -> acc + v } / n

        val eps = Value(1e-5)
        val stdInv = (variance + eps).pow(-0.5)

        return input.mapIndexed { i, x ->
            val normalized = (x - mean) * stdInv
            val scaled = normalized * weight[i]
            biasParams?.get(i)?.let { scaled + it } ?: scaled
        }.toTypedArray()
    }

    fun parameters(): List<Value> {
        return weight.toList() + (biasParams?.toList() ?: emptyList())
    }
}