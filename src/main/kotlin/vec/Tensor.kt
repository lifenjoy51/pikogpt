package vec

import RandomGaussian

/**
 * N-차원 텐서. 스칼라 `Value` 기반 autodiff와 달리 **데이터만 담는 얇은 홀더**.
 *
 * - [shape] 각 축의 크기. 예: `intArrayOf(2, 48)`은 2행 48열.
 * - [data]  크기 `shape.reduce(Int::times)`의 1차원 `FloatArray` (row-major).
 * - [grad]  역전파 시에만 지연 할당. 순전파 전용(평가/샘플링) 경로에서는 null로 남겨 메모리를 아낀다.
 *
 * 계산 그래프나 autograd 추적은 일부러 두지 않았다. 각 Layer/Op가 forward와 backward를
 * **명시적인 함수**로 구현해 chain rule이 코드에 그대로 드러나게 한다.
 */
class Tensor(
    val shape: IntArray,
    val data: FloatArray = FloatArray(shape.productInt()),
) {
    var grad: FloatArray? = null

    /** 원소의 총 개수. */
    val numel: Int get() = data.size

    /** 2차원 텐서의 행 수 ([shape]이 >= 2일 때 shape[0]). */
    val rows: Int get() = shape[0]

    /** 2차원 텐서의 열 수 (shape의 마지막 축). */
    val cols: Int get() = shape[shape.size - 1]

    /** [i, j] 접근 (2D row-major). */
    operator fun get(i: Int, j: Int): Float = data[i * cols + j]

    /** [i, j] 설정. */
    operator fun set(i: Int, j: Int, v: Float) {
        data[i * cols + j] = v
    }

    /** grad 배열을 지연 할당하고 반환. 이미 있으면 그대로. */
    fun gradOrAlloc(): FloatArray {
        val g = grad ?: FloatArray(data.size).also { grad = it }
        return g
    }

    /** 모든 grad 원소를 0으로. 옵티마이저 step 전에 호출. */
    fun zeroGrad() {
        grad?.fill(0.0f)
    }

    override fun toString(): String =
        "Tensor(shape=${shape.contentToString()}, numel=$numel, hasGrad=${grad != null})"
}

/** 크기가 0이 아닌 IntArray의 모든 원소를 곱한다. shape이 비어 있으면 1 (스칼라). */
internal fun IntArray.productInt(): Int = if (isEmpty()) 1 else reduce(Int::times)

// ---- 초기화 헬퍼 ----

/**
 * 평균 0, 표준편차 [std]의 가우시안 분포로 원소를 채운 텐서.
 * 기존 `RandomGaussian`을 그대로 재사용.
 */
fun tensorGaussian(shape: IntArray, std: Float = 0.02f): Tensor {
    val t = Tensor(shape)
    for (i in t.data.indices) {
        t.data[i] = (RandomGaussian.next() * std).toFloat()
    }
    return t
}

/** 0으로 채운 텐서. */
fun tensorZeros(shape: IntArray): Tensor = Tensor(shape)

/** 1로 채운 텐서. */
fun tensorOnes(shape: IntArray): Tensor =
    Tensor(shape).also { it.data.fill(1.0f) }

// ---- 간단한 재성형/전치 유틸 (2D 전용) ----

/** 2차원 텐서의 전치. 새 Tensor 반환. (M, N) → (N, M). */
fun Tensor.transpose2D(): Tensor {
    require(shape.size == 2) { "transpose2D는 2차원 텐서에만 사용: shape=${shape.contentToString()}" }
    val m = rows
    val n = cols
    val out = Tensor(intArrayOf(n, m))
    for (i in 0 until m) for (j in 0 until n) {
        out.data[j * m + i] = data[i * n + j]
    }
    return out
}
