package train

import Value
import kotlin.math.pow
import kotlin.math.sqrt

// AdamW 옵티마이저
class AdamW(
    private val params: List<Value>,
    private var lr: Float = 1e-3f,
    private val beta1: Float = 0.9f,
    private val beta2: Float = 0.999f,
    private val weightDecay: Float = 0.01f,
    private val eps: Float = 1e-8f
) {
    private val m = mutableMapOf<Value, Float>() // 1차 모멘트
    private val v = mutableMapOf<Value, Float>() // 2차 모멘트
    private var t = 0

    init {
        params.forEach { param ->
            m[param] = 0.0f
            v[param] = 0.0f
        }
    }

    fun step() {
        t++

        params.forEach { param ->
            if (param.grad == 0.0f) return@forEach

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
        params.forEach { it.grad = 0.0f }
    }
    
    fun updateLearningRate(newLr: Float) {
        lr = newLr
    }
}