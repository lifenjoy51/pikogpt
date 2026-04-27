package vec.layer

import vec.Tensor
import vec.tensorGaussian

/**
 * 토큰 / 위치 임베딩 테이블.
 *
 * 저장: `weight` shape `[vocabSize, embedDim]`.
 *
 *   Forward (lookup):
 *     ids: IntArray(T)  →  out[T, embedDim]
 *     out[t, :] = weight[ids[t], :]
 *
 *   Backward (grad scatter):
 *     gy[T, embedDim]을 받아 weight.grad[ids[t], :] += gy[t, :] 로 누적.
 *     같은 토큰이 여러 번 나오면 grad가 중복 누적되는데 이게 정확한 semantics.
 *
 * 입력은 grad가 필요 없는 정수 인덱스이므로 반환값(입력 grad)은 없다.
 * (VecPikoGPT 단에서 "embedding은 입력 grad 없음" 관례 유지.)
 */
class VecEmbeddingTable(
    val vocabSize: Int,
    val embedDim: Int,
) {
    val weight: Tensor = tensorGaussian(intArrayOf(vocabSize, embedDim), std = 0.02f)

    /** backward 재사용용 — 어떤 인덱스들로 lookup 했는지 기억. */
    private var cachedIds: IntArray? = null

    fun forward(ids: IntArray): Tensor {
        val t = ids.size
        val out = Tensor(intArrayOf(t, embedDim))
        for (i in 0 until t) {
            val id = ids[i]
            require(id in 0 until vocabSize) { "토큰 id 범위 밖: $id" }
            val rowStart = id * embedDim
            val outStart = i * embedDim
            for (d in 0 until embedDim) {
                out.data[outStart + d] = weight.data[rowStart + d]
            }
        }
        cachedIds = ids
        return out
    }

    /** gy[T, embedDim] 을 받아 weight.grad에 scatter-accumulate. 반환 없음. */
    fun backward(gy: Tensor) {
        val ids = cachedIds ?: error("forward 없이 backward 호출")
        require(gy.rows == ids.size && gy.cols == embedDim)
        val g = weight.gradOrAlloc()
        for (i in ids.indices) {
            val id = ids[i]
            val rowStart = id * embedDim
            val srcStart = i * embedDim
            for (d in 0 until embedDim) {
                g[rowStart + d] += gy.data[srcStart + d]
            }
        }
    }

    fun parameters(): List<Tensor> = listOf(weight)
}
