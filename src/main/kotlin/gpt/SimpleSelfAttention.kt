package gpt

import Value
import kotlin.math.sqrt

// 간단한 Self-Attention 구현 (단일 헤드 버전)
class SimpleSelfAttention(private val config: GPTConfig) {
    private val headDim = config.nEmbd / config.nHead

    // Query, Key, Value 프로젝션
    private val qProj = Linear(config.nEmbd, config.nEmbd, config.bias)
    private val kProj = Linear(config.nEmbd, config.nEmbd, config.bias)
    private val vProj = Linear(config.nEmbd, config.nEmbd, config.bias)
    private val outProj = Linear(config.nEmbd, config.nEmbd, config.bias)

    fun forward(x: Array<Array<Value>>): Array<Array<Value>> {
        val seqLen = x.size
        val batchSize = 1 // 단순화를 위해 배치 크기 1로 고정

        // Q, K, V 계산
        val queries = x.map { qProj.forward(it) }.toTypedArray()
        val keys = x.map { kProj.forward(it) }.toTypedArray()
        val values = x.map { vProj.forward(it) }.toTypedArray()

        // Attention scores 계산 (간단한 버전)
        val scale = Value(1.0 / sqrt(headDim.toDouble()))
        val scores = Array(seqLen) { i ->
            Array(seqLen) { j ->
                if (j <= i) { // Causal mask
                    var score = Value(0.0)
                    for (k in 0 until config.nEmbd) {
                        score = score + queries[i][k] * keys[j][k]
                    }
                    score * scale
                } else {
                    Value(-1e9) // 마스킹
                }
            }
        }

        // Softmax (행 단위)
        val attentionWeights = scores.map { row ->
            val maxVal = row.maxByOrNull { it.data } ?: Value(0.0)
            val expScores = row.map { (it - maxVal).exp() }.toTypedArray()
            val sumExp = expScores.reduce { acc, v -> acc + v }
            expScores.map { it / sumExp }.toTypedArray()
        }.toTypedArray()

        // Attention 적용
        val output = Array(seqLen) { i ->
            Array(config.nEmbd) { k ->
                var sum = Value(0.0)
                for (j in 0 until seqLen) {
                    sum = sum + attentionWeights[i][j] * values[j][k]
                }
                sum
            }
        }

        // 출력 프로젝션
        return output.map { outProj.forward(it) }.toTypedArray()
    }

    fun parameters(): List<Value> {
        return qProj.parameters() + kProj.parameters() +
                vProj.parameters() + outProj.parameters()
    }
}