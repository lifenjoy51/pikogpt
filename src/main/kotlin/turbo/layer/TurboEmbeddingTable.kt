package turbo.layer

import turbo.TurboTensor
import turbo.turboTensorGaussian

/**
 * 토큰/위치 임베딩 테이블 (lookup + scatter-add backward). Phase 0은 vec와 동일.
 */
class TurboEmbeddingTable(
    val vocabSize: Int,
    val embedDim: Int,
) {
    val weight: TurboTensor = turboTensorGaussian(intArrayOf(vocabSize, embedDim), std = 0.02f)

    private var cachedIds: IntArray? = null

    fun forward(ids: IntArray): TurboTensor {
        val t = ids.size
        val out = TurboTensor(intArrayOf(t, embedDim))
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

    fun backward(gy: TurboTensor) {
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

    fun parameters(): List<TurboTensor> = listOf(weight)
}
