package gpt

import Value

class LayerNorm(
    ndim: Int,
    bias: Boolean
) {
    private val weight = Array(ndim) { Value(1.0f) } // gain
    private val biasParams = if (bias) Array(ndim) { Value(0.0f) } else null // bias

    fun forward(input: Array<Value>): Array<Value> {
        val n = input.size.toFloat()
        val mean = input.reduce { acc, v -> acc + v } / n
        val variance = input.map { x -> (x - mean).pow(2.0f) }
            .reduce { acc, v -> acc + v } / n

        val eps = Value(1e-5f)
        val stdInv = (variance + eps).pow(-0.5f)

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