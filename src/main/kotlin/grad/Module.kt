package grad

import Value

abstract class Module {

    fun zeroGrad() {
        for (p in parameters()) {
            p.grad = 0.0f
        }
    }

    abstract fun parameters(): List<Value>
}