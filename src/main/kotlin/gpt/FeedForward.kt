package gpt

import Value

class FeedForward(private val config: GPTConfig) {
    private val cFc = Linear(config.nEmbd, 4 * config.nEmbd, config.bias)
    private val cProj = Linear(4 * config.nEmbd, config.nEmbd, config.bias)

    fun forward(x: Array<Value>): Array<Value> {
        var output = cFc.forward(x)
        // Value의 gelu 메서드 호출
        output = output.map { it.gelu() }.toTypedArray()
        output = cProj.forward(output)
        return output
    }

    fun parameters(): List<Value> {
        return cFc.parameters() + cProj.parameters()
    }
}