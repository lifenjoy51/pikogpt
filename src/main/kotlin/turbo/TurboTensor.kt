package turbo

import RandomGaussian

/**
 * turbo 백엔드의 N-차원 텐서. vec.Tensor와 알고리즘 동일 (Phase 0 동등성 보장).
 * 데이터는 row-major FloatArray, grad는 학습 시에만 지연 할당.
 *
 * Phase 4에서 dtype/bf16 분기를 별도 필드로 도입한다 (지금은 단일 fp32 경로만).
 */
class TurboTensor(
    @JvmField val shape: IntArray,
    @JvmField val data: FloatArray = FloatArray(shape.productInt()),
) {
    @JvmField var grad: FloatArray? = null

    val numel: Int get() = data.size
    val rows: Int get() = shape[0]
    val cols: Int get() = shape[shape.size - 1]

    operator fun get(i: Int, j: Int): Float = data[i * cols + j]
    operator fun set(i: Int, j: Int, v: Float) {
        data[i * cols + j] = v
    }

    fun gradOrAlloc(): FloatArray {
        val g = grad ?: FloatArray(data.size).also { grad = it }
        return g
    }

    fun zeroGrad() {
        grad?.fill(0.0f)
    }

    override fun toString(): String =
        "TurboTensor(shape=${shape.contentToString()}, numel=$numel, hasGrad=${grad != null})"
}

internal fun IntArray.productInt(): Int = if (isEmpty()) 1 else reduce(Int::times)

fun turboTensorGaussian(shape: IntArray, std: Float = 0.02f): TurboTensor {
    val t = TurboTensor(shape)
    for (i in t.data.indices) {
        t.data[i] = (RandomGaussian.next() * std).toFloat()
    }
    return t
}

fun turboTensorZeros(shape: IntArray): TurboTensor = TurboTensor(shape)

fun turboTensorOnes(shape: IntArray): TurboTensor =
    TurboTensor(shape).also { it.data.fill(1.0f) }

fun TurboTensor.transpose2D(): TurboTensor {
    require(shape.size == 2) { "transpose2D는 2차원 텐서에만 사용: shape=${shape.contentToString()}" }
    val m = rows
    val n = cols
    val out = TurboTensor(intArrayOf(n, m))
    for (i in 0 until m) for (j in 0 until n) {
        out.data[j * m + i] = data[i * n + j]
    }
    return out
}
