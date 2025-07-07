package train

import Value
import kotlin.math.pow
import kotlin.math.sqrt

// AdamW 옵티마이저
class AdamW(
    private val params: List<Value>,
    private val lr: Double = 1e-3,
    private val beta1: Double = 0.9,
    private val beta2: Double = 0.999,
    private val weightDecay: Double = 0.01,
    private val eps: Double = 1e-8
) {
    private val m = mutableMapOf<Value, Double>() // 1차 모멘트
    private val v = mutableMapOf<Value, Double>() // 2차 모멘트
    private var t = 0

    init {
        params.forEach { param ->
            m[param] = 0.0
            v[param] = 0.0
        }
    }

    fun step() {
        t++

        params.forEach { param ->
            if (param.grad == 0.0) return@forEach

            // Weight decay
            if (weightDecay > 0) {
                param.data -= lr * weightDecay * param.data
            }

            // Adam 업데이트
            val grad = param.grad
            m[param] = beta1 * m[param]!! + (1 - beta1) * grad
            v[param] = beta2 * v[param]!! + (1 - beta2) * grad * grad

            // Bias correction
            val mHat = m[param]!! / (1 - beta1.pow(t))
            val vHat = v[param]!! / (1 - beta2.pow(t))

            // 파라미터 업데이트
            param.data -= lr * mHat / (sqrt(vHat) + eps)
        }
    }

    fun zeroGrad() {
        params.forEach { it.grad = 0.0 }
    }
}