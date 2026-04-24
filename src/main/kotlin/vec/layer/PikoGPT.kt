package vec.layer

import gpt.GPTConfig
import vec.Tensor

/**
 * 벡터화 백엔드의 PikoGPT.
 *
 *   Forward:
 *     x[T] (토큰 ids)
 *       → tokEmb[T, C] + posEmb[T, C]     (원소별 덧셈)
 *       → stacked TransformerBlock(H, C)
 *       → final LayerNorm
 *       → lmHead (Linear C → V)
 *       → logits[T, V]
 *
 *   Backward: 위의 정확한 역순. Embedding은 scatter만 하고 입력 grad는 반환 안 함.
 */
class PikoGPT(val config: GPTConfig) {
    val tokenEmbedding = EmbeddingTable(config.vocabSize, config.nEmbd)
    val positionEmbedding = EmbeddingTable(config.blockSize, config.nEmbd)
    val blocks: Array<TransformerBlock> = Array(config.nLayer) {
        TransformerBlock(config.nEmbd, config.nHead, config.bias)
    }
    val finalLayerNorm = LayerNorm(config.nEmbd, config.bias)
    val lmHead = Linear(config.nEmbd, config.vocabSize, useBias = false)

    // backward에서 재사용할 토큰/위치 id (embedding backward가 필요)
    private var cachedTokenIds: IntArray? = null
    private var cachedPositionIds: IntArray? = null

    fun forward(tokenIds: IntArray): Tensor {
        val t = tokenIds.size
        require(t <= config.blockSize) { "시퀀스 길이 $t > blockSize ${config.blockSize}" }

        val positionIds = IntArray(t) { it }
        cachedTokenIds = tokenIds
        cachedPositionIds = positionIds

        // 1) 임베딩 덧셈
        val tokEmb = tokenEmbedding.forward(tokenIds)             // [T, C]
        val posEmb = positionEmbedding.forward(positionIds)       // [T, C]
        var x = addTensors(tokEmb, posEmb)                        // [T, C]

        // 2) 블록 스택
        for (block in blocks) {
            x = block.forward(x)
        }

        // 3) 최종 LayerNorm → lmHead
        x = finalLayerNorm.forward(x)
        return lmHead.forward(x)                                   // [T, V]
    }

    /**
     * logits에 대한 상류 기울기를 받아 모델 전체 backward를 수행한다.
     * 파라미터 grad만 누적하고 별도 반환값은 없다 (입력은 정수 토큰이라 grad 없음).
     */
    fun backward(gLogits: Tensor) {
        // 역순으로 체인 룰
        val dAfterLn = lmHead.backward(gLogits)                    // [T, C]
        val dAfterBlocks = finalLayerNorm.backward(dAfterLn)       // [T, C]

        var g = dAfterBlocks
        for (block in blocks.reversed()) {
            g = block.backward(g)
        }

        // 임베딩 덧셈의 backward: grad가 tokEmb/posEmb에 동일하게 전달됨
        tokenEmbedding.backward(g)
        positionEmbedding.backward(g)
    }

    fun parameters(): List<Tensor> {
        val list = mutableListOf<Tensor>()
        list += tokenEmbedding.parameters()
        list += positionEmbedding.parameters()
        blocks.forEach { list += it.parameters() }
        list += finalLayerNorm.parameters()
        list += lmHead.parameters()
        return list
    }

    fun zeroGrad() {
        parameters().forEach { it.zeroGrad() }
    }

    private fun addTensors(a: Tensor, b: Tensor): Tensor {
        require(a.numel == b.numel)
        val out = Tensor(a.shape.copyOf())
        for (i in out.data.indices) out.data[i] = a.data[i] + b.data[i]
        return out
    }
}
