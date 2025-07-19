package gpt

import Value

// Transformer 블록
class TransformerBlock(config: GPTConfig) {
    private val ln1 = LayerNorm(config.nEmbd, config.bias)
    private val attn = SimpleSelfAttention(config)
    private val ln2 = LayerNorm(config.nEmbd, config.bias)
    private val mlp = MLP(config)

    fun forward(x: Array<Array<Value>>): Array<Array<Value>> {
        // x + self.attn(self.ln_1(x))
        val normalized1 = x.map { ln1.forward(it) }.toTypedArray()
        val attnOut = attn.forward(normalized1)
        val x1 = x.zip(attnOut) { xi, attni ->
            xi.zip(attni) { a, b -> a + b }.toTypedArray()
        }.toTypedArray()

        // x + self.mlp(self.ln_2(x))
        val normalized2 = x1.map { ln2.forward(it) }.toTypedArray()
        val mlpOut = normalized2.map { mlp.forward(it) }.toTypedArray()
        val x2 = x1.zip(mlpOut) { xi, mlpi ->
            xi.zip(mlpi) { a, b -> a + b }.toTypedArray()
        }.toTypedArray()

        return x2
    }

    fun parameters(): List<Value> {
        return ln1.parameters() + attn.parameters() + ln2.parameters() + mlp.parameters()
    }
}