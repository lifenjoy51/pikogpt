package gpt

import RandomGaussian
import Value
import kotlin.random.Random

class Linear(
    inFeatures: Int,
    private val outFeatures: Int,
    bias: Boolean = true
) {
    // 표준 라이브러리의 nextGaussian() 사용
    private val weight = Array(outFeatures) {
        Array(inFeatures) { Value((RandomGaussian.next() * 0.02).toFloat()) }
    }
    private val biasParams = if (bias) Array(outFeatures) { Value(0.0f) } else null

    fun forward(input: Array<Value>): Array<Value> {
        return Array(outFeatures) { i ->
            val weightedSum = input.zip(weight[i]) { inp, w -> inp * w }
                .reduce { acc, v -> acc + v }

            biasParams?.get(i)?.let { weightedSum + it } ?: weightedSum
        }
    }

    fun parameters(): List<Value> {
        // flatten과 elvis 연산자를 사용하여 더 간결하게 변경
        return weight.flatten() + (biasParams?.toList() ?: emptyList())
    }
}