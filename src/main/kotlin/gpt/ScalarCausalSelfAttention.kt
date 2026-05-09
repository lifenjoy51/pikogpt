package gpt

import Value
import kotlin.math.sqrt

/**
 * Causal Multi-Head Self-Attention (단순화된 구현).
 *
 * Transformer의 핵심. 시퀀스의 각 위치가 다른 위치들을 "보면서" 표현을 갱신합니다. 자기회귀
 * 언어 모델에선 미래 위치를 보지 못하도록 causal mask를 적용합니다.
 *
 * `forward()`의 본문은 5개 helper의 파이프라인 — 각 단계가 한눈에 보이도록 분해되어 있습니다:
 *   1. [projectQkv]            — Q, K, V 행렬 생성
 *   2. [attentionScores]       — scaled dot-product + causal mask
 *   3. [Matrix.softmaxRows]    — row-wise softmax (수치 안정화 max-trick 포함)
 *   4. [weightedSum]           — attention 가중치와 V의 가중 합
 *   5. [outputAndDropout]      — 출력 프로젝션 + dropout
 *
 * @param modelConfig GPT 모델 설정 (어텐션 헤드 수, 임베딩 차원 등)
 */
class ScalarCausalSelfAttention(private val modelConfig: GPTConfig) {
    /** 각 어텐션 헤드의 차원 (embedding_dim / num_heads). */
    private val attentionHeadDimension = modelConfig.embeddingDimension / modelConfig.numberOfAttentionHeads

    private val queryProjection = ScalarLinear(modelConfig.embeddingDimension, modelConfig.embeddingDimension, modelConfig.useBias)
    private val keyProjection = ScalarLinear(modelConfig.embeddingDimension, modelConfig.embeddingDimension, modelConfig.useBias)
    private val valueProjection = ScalarLinear(modelConfig.embeddingDimension, modelConfig.embeddingDimension, modelConfig.useBias)

    /** 어텐션 결과를 다시 임베딩 차원으로 사영. */
    private val outputProjection = ScalarLinear(modelConfig.embeddingDimension, modelConfig.embeddingDimension, modelConfig.useBias)

    /** attention dropout. 학습 안정화. */
    private val attentionDropout = ScalarDropout(modelConfig.dropoutProbability)

    /**
     * 5단계 파이프라인. 각 helper가 한 단계를 담당.
     *
     *   1. Q, K, V 사영
     *   2. Q·K^T / √d_k + causal mask  → scores
     *   3. row-wise softmax            → attention weights
     *   4. weights · V                 → context vectors
     *   5. output projection + dropout
     *
     * @param input [tokens, embed_dim]
     * @return 같은 형태의 출력
     */
    fun forward(input: Matrix): Matrix {
        val (queries, keys, values) = projectQkv(input)
        val scores = attentionScores(queries, keys)
        val weights = scores.softmaxRows()
        val context = weightedSum(weights, values)
        return outputAndDropout(context)
    }

    /**
     * 단계 1: Q, K, V 행렬 생성.
     *
     * 입력의 각 토큰 임베딩에 세 개의 학습된 선형 변환을 적용해 Query, Key, Value를 얻습니다.
     * Q는 "이 토큰이 무엇을 찾는지", K는 "이 토큰이 무엇을 제공하는지", V는 "실제로 전달되는
     * 정보"의 역할.
     */
    private fun projectQkv(input: Matrix): Triple<Matrix, Matrix, Matrix> {
        val queries = input.mapRows { tokenEmbedding -> queryProjection.forward(tokenEmbedding) }
        val keys = input.mapRows { tokenEmbedding -> keyProjection.forward(tokenEmbedding) }
        val values = input.mapRows { tokenEmbedding -> valueProjection.forward(tokenEmbedding) }
        return Triple(queries, keys, values)
    }

    /**
     * 단계 2: scaled dot-product attention scores + causal mask.
     *
     *   scores[i, j] = (Q[i] · K[j]) / √d_k                if j ≤ i
     *                = -∞ (very large negative)            if j > i  ← causal mask
     *
     * √d_k 스케일링은 dot product의 분산이 d_k에 비례해 커지는 것을 방지 (softmax가
     * 한 점에 몰리는 것을 막음).
     *
     * causal mask는 미래 위치에 매우 작은 값을 넣어, softmax 후 가중치가 사실상 0이 되도록.
     */
    private fun attentionScores(queries: Matrix, keys: Matrix): Matrix {
        val tokenCount = queries.rows
        val attentionScale = Value(1.0f / sqrt(attentionHeadDimension.toFloat()))

        return Matrix.fromArray(Array(tokenCount) { queryIndex ->
            Array(tokenCount) { keyIndex ->
                if (keyIndex <= queryIndex) {
                    // Q[queryIndex] · K[keyIndex]
                    var dotProduct = Value.ZERO
                    for (embeddingIndex in 0 until modelConfig.embeddingDimension) {
                        val q = queries[queryIndex][embeddingIndex]
                        val k = keys[keyIndex][embeddingIndex]
                        dotProduct += q * k
                    }
                    dotProduct * attentionScale
                } else {
                    // 미래 위치는 매우 작은 값 → softmax 후 거의 0
                    Value.MIN
                }
            }
        })
    }

    /**
     * 단계 4: attention 가중치를 V에 적용한 가중 합.
     *
     *   context[i, d] = Σ_j weights[i, j] * V[j, d]
     *
     * 각 query 위치에 대해 모든 (causal하게 허용된) value 벡터의 가중 평균을 계산.
     */
    private fun weightedSum(weights: Matrix, values: Matrix): Matrix {
        val tokenCount = weights.rows
        val embedDim = modelConfig.embeddingDimension
        val output = Array(tokenCount) { queryIndex ->
            Array(embedDim) { embeddingIndex ->
                var weightedSum = Value.ZERO
                for (keyIndex in 0 until tokenCount) {
                    val w = weights[queryIndex][keyIndex]
                    val v = values[keyIndex][embeddingIndex]
                    weightedSum += w * v
                }
                weightedSum
            }
        }
        return Matrix.fromArray(output)
    }

    /**
     * 단계 5: 출력 프로젝션 + dropout.
     *
     * 어텐션의 context 벡터를 다시 임베딩 차원으로 사영하고 dropout을 적용해 최종 출력 생성.
     */
    private fun outputAndDropout(context: Matrix): Matrix {
        val projected = context.mapRows { contextVector -> outputProjection.forward(contextVector) }
        return attentionDropout.forward(projected)
    }

    /**
     * 학습 가능한 파라미터 — Q/K/V 사영 + 출력 사영의 가중치/편향.
     */
    fun parameters(): List<Value> {
        return queryProjection.parameters() + keyProjection.parameters() +
            valueProjection.parameters() + outputProjection.parameters()
    }
}
