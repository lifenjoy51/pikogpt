package gpt

import Value

class PikoGPT(val config: GPTConfig) {
    private val tokenEmbedding = Array(config.vocabSize) {
        Array(config.nEmbd) { Value(RandomGaussian.next() * 0.02) }
    }
    private val positionEmbedding = Array(config.blockSize) {
        Array(config.nEmbd) { Value(RandomGaussian.next() * 0.02) }
    }
    private val blocks = Array(config.nLayer) { TransformerBlock(config) }
    private val lnF = LayerNorm(config.nEmbd, config.bias)
    private val lmHead = Linear(config.nEmbd, config.vocabSize, false)

    fun forward(tokenIds: IntArray): Array<Array<Value>> {
        val seqLen = tokenIds.size
        val embeddings = Array(seqLen) { i ->
            val tokEmb = tokenEmbedding[tokenIds[i]]
            val posEmb = positionEmbedding[i]
            tokEmb.zip(posEmb) { t, p -> t + p }.toTypedArray()
        }

        var x = blocks.fold(embeddings) { current, block -> block.forward(current) }
        x = x.map { lnF.forward(it) }.toTypedArray()

        return x.map { lmHead.forward(it) }.toTypedArray()
    }

    fun parameters(): List<Value> {
        // flatMap과 + 연산자를 사용하여 더 간결하게 변경
        return tokenEmbedding.flatten() +
                positionEmbedding.flatten() +
                blocks.flatMap { it.parameters() } +
                lnF.parameters() + // gpt.LayerNorm 파라미터 추가
                lmHead.parameters()
    }
}