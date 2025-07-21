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
class TransformerBlock(config: GPTConfig) {
    /** 첫 번째 Layer Normalization (어텐션 전) */
    private val ln1 = LayerNorm(config.nEmbd, config.bias)

    /** Self-Attention 메커니즘 */
    private val attn = SimpleSelfAttention(config)

    /** 두 번째 Layer Normalization (MLP 전) */
    private val ln2 = LayerNorm(config.nEmbd, config.bias)

    /** Multi-Layer Perceptron (Feed-Forward Network) */
    private val mlp = MLP(config)

    /**
     * Transformer 블록 순전파
     *
     * Pre-Norm 구조를 사용하여 각 서브 레이어 전에 Layer Normalization을 적용합니다.
     * 잔여 연결을 통해 그래디언트 흐름을 개선하고 훈련 안정성을 향상시킵니다.
     *
     * @param x 입력 시퀀스
     * @return 변환된 시퀀스
     */
    fun forward(x: Sequence): Sequence {
        // 첫 번째 서브레이어: x + self.attn(self.ln_1(x))
        val normalized1 = x.map { ln1.forward(it) }
        val attnOut = attn.forward(normalized1)
        val x1 = x.zipWith(attnOut) { a, b -> a + b }

        // 두 번째 서브레이어: x + self.mlp(self.ln_2(x))
        val normalized2 = x1.map { ln2.forward(it) }
        val mlpSequence = normalized2.map { mlp.forward(it) }
        val x2 = x1.zipWith(mlpSequence) { a, b -> a + b }

        return x2
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