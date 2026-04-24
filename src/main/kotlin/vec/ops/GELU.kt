package vec.ops

import vec.Tensor
import kotlin.math.PI
import kotlin.math.sqrt
import kotlin.math.tanh

/**
 * GELU (Gaussian Error Linear Unit) — Transformer FFN의 표준 활성화.
 *
 * 원래 정의:  GELU(x) = x · Φ(x)  (Φ는 표준 정규 CDF)
 * tanh 근사:  GELU(x) ≈ 0.5 · x · ( 1 + tanh( √(2/π) · ( x + 0.044715 · x^3 ) ) )
 *
 * Forward는 원소별 연산. Backward도 원소별이라 간단하다.
 *
 *   let a = √(2/π), k = 0.044715
 *   inner(x) = a * (x + k * x^3)
 *   t(x)     = tanh(inner(x))
 *   GELU(x)  = 0.5 * x * (1 + t(x))
 *
 *   dGELU/dx = 0.5 * (1 + t) + 0.5 * x * (1 - t^2) * inner'(x)
 *   inner'(x) = a * (1 + 3k * x^2)
 *
 * 이 파일은 forward와 backward를 쌍으로 제공한다. forward가 반환한 Tensor와 원본 x를
 * backward에 함께 넘기는 게 아니라, **backward는 원본 x만 필요**하다. 재계산이 아깝지 않고
 * 캐시 관리가 단순해지는 쪽을 택했다.
 */
private const val GELU_A: Float = 0.7978845608028654f   // √(2/π)
private const val GELU_K: Float = 0.044715f

fun gelu(x: Tensor): Tensor {
    val out = Tensor(x.shape.copyOf())
    for (i in x.data.indices) {
        val v = x.data[i]
        val inner = GELU_A * (v + GELU_K * v * v * v)
        out.data[i] = 0.5f * v * (1.0f + tanh(inner.toDouble()).toFloat())
    }
    return out
}

/**
 * GELU backward. `x`는 forward 입력, `gyData`는 출력에 대한 기울기, 반환은 입력 기울기.
 */
fun geluBackward(x: Tensor, gyData: FloatArray): FloatArray {
    require(gyData.size == x.numel)
    val dx = FloatArray(x.numel)
    for (i in x.data.indices) {
        val v = x.data[i]
        val inner = GELU_A * (v + GELU_K * v * v * v)
        val t = tanh(inner.toDouble()).toFloat()
        val innerPrime = GELU_A * (1.0f + 3.0f * GELU_K * v * v)
        val localDerivative = 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * innerPrime
        dx[i] = gyData[i] * localDerivative
    }
    return dx
}

// tanh 근사 상수 참고용 — √(2/π) 를 소수 16자리까지
@Suppress("unused")
private fun tanhApproxConstantRef(): Double = sqrt(2.0 / PI)
