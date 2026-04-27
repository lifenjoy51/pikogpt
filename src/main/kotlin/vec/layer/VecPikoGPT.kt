package vec.layer

import gpt.GPTConfig
import vec.Tensor
import vec.ops.matmul
import vec.ops.matmulBackward
import vec.transpose2D

/**
 * 벡터화 백엔드의 VecPikoGPT.
 *
 *   Forward:
 *     x[T] (토큰 ids)
 *       → tokEmb[T, C] + posEmb[T, C]     (원소별 덧셈)
 *       → embedding dropout (표준 GPT-2 스타일)
 *       → stacked VecTransformerBlock(H, C)
 *       → final VecLayerNorm
 *       → lmHead (VecLinear C → V) — `config.tieWeights=true`면 tokEmb 가중치 재사용
 *       → logits[T, V]
 *
 *   Backward: 위의 정확한 역순. Embedding은 scatter만 하고 입력 grad는 반환 안 함.
 *
 * **Weight tying** (config.tieWeights=true):
 *   token_embedding.weight를 lm_head 가중치로 재사용. lm_head는 별도 VecLinear를 만들지
 *   않고 forward에서 `matmul(x, tokenEmbedding.weight^T)`를 직접 계산한다.
 *   - 파라미터 절약: vocab × dim (작은 모델일수록 비중 큼)
 *   - 임베딩이 forward(lookup) + backward(matmul) 두 경로에서 grad 받음 → 학습 신호 강화
 *   - GPT-2/GPT-3/Llama 등 거의 모든 production LM의 표준 관행
 *   - 기존 untied ckpt와 직렬화 호환 안 됨 (param 수 다름).
 */
class VecPikoGPT(val config: GPTConfig) {
    /** RoPE 사용 시 학습 가능 position embedding 불필요 — Q·K 회전이 위치 정보를 주입. */
    private val useRoPE: Boolean = config.positionEncoding.equals("rope", ignoreCase = true)

    val tokenEmbedding = VecEmbeddingTable(config.vocabularySize, config.embeddingDimension)
    /** learned position embedding. RoPE 모드에서는 null (parameters에서도 제외). */
    val positionEmbedding: VecEmbeddingTable? =
        if (useRoPE) null else VecEmbeddingTable(config.maxSequenceLength, config.embeddingDimension)
    val embeddingDropout = VecDropout(config.dropoutProbability)
    val blocks: Array<VecTransformerBlock> = Array(config.numberOfLayers) {
        VecTransformerBlock(config.embeddingDimension, config.numberOfAttentionHeads, config.useBias, config.dropoutProbability, config.mlpActivation, config.positionEncoding)
    }
    val finalLayerNorm = VecLayerNorm(config.embeddingDimension, config.useBias)

    /** untied 모드일 때만 별도 lm_head VecLinear. tied 모드면 null이고 forward에서 직접 계산. */
    val lmHead: VecLinear? =
        if (config.tieWeights) null
        else VecLinear(config.embeddingDimension, config.vocabularySize, useBias = false)

    // backward에서 재사용할 토큰/위치 id + lmHead 입력 (tied 경로용)
    private var cachedTokenIds: IntArray? = null
    private var cachedPositionIds: IntArray? = null
    private var cachedHeadInput: Tensor? = null

    fun forward(tokenIds: IntArray): Tensor {
        val t = tokenIds.size
        require(t <= config.maxSequenceLength) { "시퀀스 길이 $t > maxSequenceLength ${config.maxSequenceLength}" }

        val positionIds = IntArray(t) { it }
        cachedTokenIds = tokenIds
        cachedPositionIds = positionIds

        // 1) 임베딩 — RoPE 모드면 token만, 아니면 token + position.
        val tokEmb = tokenEmbedding.forward(tokenIds)             // [T, C]
        var x = if (positionEmbedding != null) {
            val posEmb = positionEmbedding.forward(positionIds)   // [T, C]
            addTensors(tokEmb, posEmb)
        } else {
            tokEmb
        }
        x = embeddingDropout.forward(x)

        // 2) 블록 스택
        for (block in blocks) {
            x = block.forward(x)
        }

        // 3) 최종 VecLayerNorm → lm_head
        x = finalLayerNorm.forward(x)
        return if (lmHead != null) {
            lmHead.forward(x)                                      // untied
        } else {
            // Tied: logits = x · tokEmb.weight^T  (x[T,C] · [C,V] = [T,V])
            cachedHeadInput = x
            matmul(x, tokenEmbedding.weight.transpose2D())         // [T, V]
        }
    }

    /**
     * logits에 대한 상류 기울기를 받아 모델 전체 backward를 수행한다.
     * 파라미터 grad만 누적하고 별도 반환값은 없다 (입력은 정수 토큰이라 grad 없음).
     */
    fun backward(gLogits: Tensor) {
        // 역순으로 체인 룰. lm_head 단계에서 tied vs untied 분기.
        val dAfterLn = if (lmHead != null) {
            lmHead.backward(gLogits)                               // [T, C]
        } else {
            // tied lm_head backward:
            //   forward: y = x · W^T (W = tokenEmbedding.weight, shape [V, C])
            //   ∂L/∂x = gLogits · W              (shape [T, C])
            //   ∂L/∂W += gLogits^T · x           (shape [V, C])
            // tokenEmbedding.weight.grad에 추가 누적 — 이후 token lookup backward의
            //   scatter-add와 합산되어 정확히 양방향 grad 합 형태로 학습 신호 받음.
            val x = cachedHeadInput ?: error("forward 없이 backward (tied head)")
            val w = tokenEmbedding.weight                          // [V, C]
            // ∂L/∂x = gLogits · W
            val dx = matmul(gLogits, w)
            // ∂L/∂W += gLogits^T · x   (matmulBackward는 a·b backward와 형태 다르므로 수동)
            val wGrad = w.gradOrAlloc()
            val v = w.rows
            val c = w.cols
            val n = gLogits.rows
            for (vv in 0 until v) {
                for (cc in 0 until c) {
                    var sum = 0.0f
                    for (nn in 0 until n) {
                        sum += gLogits.data[nn * v + vv] * x.data[nn * c + cc]
                    }
                    wGrad[vv * c + cc] += sum
                }
            }
            dx
        }

        val dAfterBlocks = finalLayerNorm.backward(dAfterLn)       // [T, C]

        var g = dAfterBlocks
        for (block in blocks.reversed()) {
            g = block.backward(g)
        }

        // embedding dropout backward → tokEmb (+ posEmb if not RoPE) grad
        g = embeddingDropout.backward(g)

        // 임베딩 덧셈의 backward: grad가 tokEmb/posEmb에 동일하게 전달됨.
        // tied 모드면 tokenEmbedding.weight.grad는 위 lm_head backward에서 누적됐던 값에
        //   여기서의 scatter-add가 추가되어 양방향 grad 합으로 정확.
        // RoPE 모드면 positionEmbedding 자체가 없음.
        tokenEmbedding.backward(g)
        positionEmbedding?.backward(g)
    }

    /**
     * 학습/추론 모드 토글. 모든 내부 VecDropout 레이어의 `training` 플래그를 설정한다.
     * 학습 루프에서는 true, 평가·샘플링에서는 false.
     */
    fun setTraining(mode: Boolean) {
        embeddingDropout.training = mode
        for (block in blocks) {
            block.attention.attnDropout.training = mode
            block.attention.residDropout.training = mode
            block.mlp.dropout.training = mode
        }
    }

    fun parameters(): List<Tensor> {
        val list = mutableListOf<Tensor>()
        list += tokenEmbedding.parameters()
        if (positionEmbedding != null) list += positionEmbedding.parameters()  // RoPE면 없음
        blocks.forEach { list += it.parameters() }
        list += finalLayerNorm.parameters()
        if (lmHead != null) list += lmHead.parameters()  // tied면 추가 안 함 (이미 tokenEmbedding으로 등록)
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
