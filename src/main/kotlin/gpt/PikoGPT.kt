package gpt

import RandomGaussian
import Value

/**
 * PikoGPT - 미니 GPT 모델 구현
 *
 * GPT(Generative Pre-trained Transformer) 아키텍처를 바탕으로 한 언어 모델입니다.
 * Transformer 블록들을 여러 개 연결하여 다음 토큰을 예측하는 자기회귀적(auto-regressive) 모델입니다.
 *
 * 모델 구조:
 * 1. Token Embedding: 단어 토큰을 벡터로 매핑
 * 2. Position Embedding: 위치 정보를 벡터로 매핑
 * 3. Transformer Blocks: Self-Attention + Feed-Forward 블록들
 * 4. Layer Normalization: 최종 정규화
 * 5. Language Model Head: 어휘 확률 분포 생성
 *
 * @param config GPT 모델의 하이퍼파라미터 설정
 */
class PikoGPT(val config: GPTConfig) {

    /** 토큰 임베딩 테이블 [vocab_size, embedding_dim] */
    private val tokenEmbedding = EmbeddingTable(config.vocabSize, config.nEmbd)

    /** 위치 임베딩 테이블 [block_size, embedding_dim] */
    private val positionEmbedding = EmbeddingTable(config.blockSize, config.nEmbd)

    /** Transformer 블록들의 리스트 */
    private val blocks = Array(config.nLayer) { TransformerBlock(config) }

    /** 최종 Layer Normalization */
    private val lnF = LayerNorm(config.nEmbd, config.bias)

    /** Language Model Head - 어휘 로짓 생성 레이어 */
    private val lmHead = Linear(config.nEmbd, config.vocabSize, false)

    /**
     * 순전파 (Forward Pass)
     *
     * 입력 토큰 시퀀스를 받아 다음 토큰에 대한 로짓 분포를 출력합니다.
     *
     * 처리 단계:
     * 1. 토큰 임베딩 + 위치 임베딩
     * 2. 여러 Transformer 블록을 통과
     * 3. 최종 Layer Normalization
     * 4. Language Model Head로 어휘 로짓 생성
     *
     * @param tokenIds 입력 토큰 ID 배열 [sequence_length]
     * @return 각 위치에서의 어휘 로짓 분포
     */
    fun forward(tokenIds: IntArray): Logits {
        val seqLen = tokenIds.size

        // 1. 임베딩 레이어: 토큰 + 위치 임베딩
        val tokenSequence = tokenEmbedding.lookup(tokenIds)
        val positionIds = IntArray(seqLen) { it } // [0, 1, 2, ..., seqLen-1]
        val positionSequence = positionEmbedding.lookup(positionIds)
        
        // 요소별 덧셈으로 결합
        var sequence = tokenSequence.zipWith(positionSequence) { t, p -> t + p }

        // 2. 모든 Transformer 블록을 순차적으로 통과
        for (block in blocks) {
            sequence = block.forward(sequence)
        }

        // 3. 최종 Layer Normalization
        sequence = sequence.map { lnF.forward(it) }

        // 4. Language Model Head로 어휘 로짓 생성
        val logitValues = sequence.values.map { lmHead.forward(it) }.toTypedArray()
        return Logits.fromArray(logitValues)
    }

    /**
     * 모델의 모든 파라미터 수집
     *
     * 옵티마이저와 그래디언트 계산을 위해 모델의 모든 학습 가능한 파라미터를 수집합니다.
     *
     * 포함되는 파라미터:
     * - 토큰 임베딩 가중치
     * - 위치 임베딩 가중치
     * - 모든 Transformer 블록의 파라미터
     * - 최종 Layer Normalization 파라미터
     * - Language Model Head 파라미터
     *
     * @return 모든 학습 가능한 Value 객체들의 리스트
     */
    fun parameters(): List<Value> {
        return tokenEmbedding.parameters() +
                positionEmbedding.parameters() +
                blocks.flatMap { it.parameters() } +
                lnF.parameters() +
                lmHead.parameters()
    }
}