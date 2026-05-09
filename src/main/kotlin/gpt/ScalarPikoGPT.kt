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
class ScalarPikoGPT(val config: GPTConfig) {

    /** 토큰 임베딩 테이블 [vocab_size, embedding_dim] */
    private val tokenEmbedding = ScalarEmbeddingTable(config.vocabularySize, config.embeddingDimension)

    /** 위치 임베딩 테이블 [block_size, embedding_dim] */
    private val positionEmbedding = ScalarEmbeddingTable(config.maxSequenceLength, config.embeddingDimension)

    /** Transformer 블록들의 리스트 */
    private val blocks = Array(config.numberOfLayers) { ScalarTransformerBlock(config) }

    /** 최종 Layer Normalization */
    private val lnF = ScalarLayerNorm(config.embeddingDimension, config.useBias)

    /** Language Model Head - 어휘 로짓 생성 레이어 */
    private val lmHead = ScalarLinear(config.embeddingDimension, config.vocabularySize, false)

    /**
     * 순전파.
     *
     * 입력 토큰 시퀀스 → 다음 토큰에 대한 로짓 분포.
     *
     * 단계:
     * 1. 토큰 임베딩 + 위치 임베딩
     * 2. 여러 Transformer 블록을 순차 통과
     * 3. 최종 Layer Normalization
     * 4. Language Model Head로 vocab 로짓 생성
     *
     * @param tokenIds 입력 토큰 ID 배열
     * @return [seqLen, vocabSize] logits 행렬
     */
    fun forward(tokenIds: IntArray): Matrix {
        val seqLen = tokenIds.size

        // 1. 임베딩 — 토큰 + 위치
        val tokenSequence = tokenEmbedding.lookup(tokenIds)
        val positionIds = IntArray(seqLen) { it }
        val positionSequence = positionEmbedding.lookup(positionIds)
        var hidden = tokenSequence.zipWith(positionSequence) { t, p -> t + p }

        // 2. Transformer 블록 스택
        for (block in blocks) {
            hidden = block.forward(hidden)
        }

        // 3. 최종 Layer Norm
        hidden = hidden.mapRows { lnF.forward(it) }

        // 4. LM Head — 각 위치에서 vocab 로짓 생성
        val logitRows = hidden.values
            .map { lmHead.forward(it) }
            .toTypedArray()
        return Matrix.fromArray(logitRows)
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