package gpt

import Value

class MLP(config: GPTConfig) {
    private val cFc = Linear(config.nEmbd, 4 * config.nEmbd, config.bias)
    private val cProj = Linear(4 * config.nEmbd, config.nEmbd, config.bias)
    private val dropout = Dropout(config.dropout)

    fun forward(x: Array<Value>): Array<Value> {
        var output = cFc.forward(x)
        // Value의 gelu 메서드 호출
        output = output.map { it.gelu() }.toTypedArray()
        output = cProj.forward(output)
        // Dropout 적용
        return dropout.forward(output.toList()).toTypedArray()
    }

    fun parameters(): List<Value> {
        return cFc.parameters() + cProj.parameters()
    }
}