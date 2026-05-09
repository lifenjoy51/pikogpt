package gpt

import Value

/**
 * Transformer 블록
 *
 * GPT의 핵심 구성 요소로, Self-Attention과 Feed-Forward Network를 결합한 블록입니다.
 * 잔여 연결(Residual Connection)과 Layer Normalization을 사용하여 안정성을 향상시킵니다.
 *
 * 블록 구조:
 * 1. Layer Norm -> Self-Attention -> Residual Connection
 * 2. Layer Norm -> MLP -> Residual Connection
 *
 * 이는 Pre-Norm 구조로, 기존 Post-Norm보다 훈련 안정성이 더 좋습니다.
 *
 * @param config GPT 모델 설정
 */
class ScalarTransformerBlock(config: GPTConfig) {
    /** 첫 번째 Layer Normalization (어텐션 전) */
    private val ln1 = ScalarLayerNorm(config.embeddingDimension, config.useBias)

    /** Self-Attention 메커니즘 */
    private val attn = ScalarCausalSelfAttention(config)

    /** 두 번째 Layer Normalization (MLP 전) */
    private val ln2 = ScalarLayerNorm(config.embeddingDimension, config.useBias)

    /** Multi-Layer Perceptron (Feed-Forward Network) */
    private val mlp = ScalarFeedForward(config)

    /**
     * Transformer 블록 순전파 (Pre-Norm).
     *
     *   h1 = x  + attn(ln1(x))
     *   y  = h1 + mlp(ln2(h1))
     *
     * 각 sub-layer 앞에 LayerNorm을 두는 Pre-Norm 구조 — Post-Norm보다 학습 안정성이 좋습니다.
     * 잔여 연결로 그래디언트가 깊은 스택을 통과해도 흐름이 유지됩니다.
     *
     * @param x 입력 [tokens, embed_dim]
     * @return 같은 형태의 출력
     */
    fun forward(x: Matrix): Matrix {
        // 첫 번째 서브레이어: x + attn(ln1(x))
        val normalized1 = x.mapRows { ln1.forward(it) }
        val attnOut = attn.forward(normalized1)
        val h1 = x.zipWith(attnOut) { a, b -> a + b }

        // 두 번째 서브레이어: h1 + mlp(ln2(h1))
        val normalized2 = h1.mapRows { ln2.forward(it) }
        val mlpOut = normalized2.mapRows { mlp.forward(it) }
        return h1.zipWith(mlpOut) { a, b -> a + b }
    }

    /**
     * Transformer 블록의 모든 파라미터 수집
     *
     * 이 블록에 포함된 모든 레이어의 학습 가능한 파라미터를 수집합니다.
     *
     * @return 모든 파라미터들의 통합 리스트
     */
    fun parameters(): List<Value> {
        return ln1.parameters() + attn.parameters() + ln2.parameters() + mlp.parameters()
    }
}