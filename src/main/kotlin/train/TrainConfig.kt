package train

import kotlinx.serialization.Serializable

/**
 * 훈련 설정 클래스
 *
 * GPT 모델 훈련에 필요한 모든 하이퍼파라미터와 설정을 정의합니다.
 * 이 설정들은 모델 아키텍처, 훈련 전략, 데이터 처리 방식 등을 제어합니다.
 */
@Serializable
data class TrainConfig(
    // I/O
    val dataPath: String = "data", // 데이터 경로
    val modelDir: String = "model",
    val evalIntervalRatio: Float = 0.01f, //50,
    val logInterval: Int = 1,
    val evalIters: Int = 1, // 평가 이터레이션 최소화
    val evalOnly: Boolean = false,
    val alwaysSaveCheckpoint: Boolean = true,
    val initFrom: String = "scratch", // 'scratch' or 'resume'
    val subDir: String? = null, // 모델 디렉토리

    // 데이터
    val dataset: String = "stories",
    val gradientAccumulationSteps: Int = 4, // 메모리의 한계로 인해 배치 크기(batchSize)를 키울 수 없을 때, 실질적인(Effective) 배치 크기를 늘리는 효과를 내는 기법입니다. Effective Batch Size = batchSize * gradientAccumulationSteps
    val batchSize: Int = 4, // 한 번의 모델 가중치 업데이트(1 스텝)에 사용되는 데이터 샘플의 수를 의미합니다.
    val blockSize: Int = 24, // 모델이 한 번의 예측을 위해 참고하는 토큰(단어)의 최대 개수를 의미합니다. 즉, 모델의 "시야" 또는 "단기 기억력"의 범위를 결정합니다.

    // 모델
    val embeddingDimension: Int = 4, // 가장 기본적인 성능 향상 방법입니다. 모델이 각 토큰을 표현하는 정보의 양을 늘려줍니다.
    val numberOfLayers: Int = 1, // 모델의 깊이를 늘려 더 복잡하고 추상적인 패턴을 학습하게 합니다.
    val numberOfHeads: Int = 1, // 어텐션 메커니즘이 한 번에 다양한 종류의 관계를 보도록 합니다. numberOfHeads는 embeddingDimension의 약수여야 합니다.
    val bias: Boolean = true,
    val dropout: Float = 0.15f, // Dropout 확률

    // 옵티마이저
    val learningRate: Float = 5e-4f,
    val maxIters: Int = 5000,
    val weightDecay: Float = 0.05f,
    val beta1: Float = 0.9f,
    val beta2: Float = 0.99f,
    val gradClip: Float = 1.0f,

    // 학습률 스케줄
    val decayLr: Boolean = true,
    val warmupRatio: Float = 0.01f, // maxIters의 10%를 warmup으로 사용
    val learningRateDecayRatio: Float = 0.8f, // maxIters의 100%까지 decay (훈련 끝까지)
    val minimumLearningRate: Float = 1e-5f
) {
    // 계산된 속성들
    val warmupIters: Int get() = (maxIters * warmupRatio).toInt()
    val learningRateDecayIterations: Int get() = (maxIters * learningRateDecayRatio).toInt()
    val evalInterval: Int get() = (maxIters * evalIntervalRatio).toInt()

    /**
     * GPT 모델의 총 파라미터 수 계산
     *
     * 모델의 모든 레이어에 포함된 파라미터 수를 상세하게 계산하여 보고합니다.
     * 이는 모델의 복잡도와 훈련 비용을 추정하는데 도움이 됩니다.
     *
     * 계산 방식:
     * 1. 임베딩 레이어: Token + Position Embedding
     * 2. Transformer 블록들: Attention + FFN + LayerNorm
     * 3. 출력 레이어: Language Model Head
     *
     * @param vocabularySize 모델의 어휘 사전 크기 (e.g., 65 for shakespeare_char)
     * @return 총 파라미터 수 (Long 타입)
     */
    fun calculateTotalParameters(vocabularySize: Int): Long {
        var totalParameters = 0L

        // --- 1. 임베딩 레이어 ---
        val tokenEmbeddingParameters = vocabularySize.toLong() * this.embeddingDimension
        val positionEmbeddingParameters = this.blockSize.toLong() * this.embeddingDimension
        totalParameters += tokenEmbeddingParameters + positionEmbeddingParameters
        println(String.format("1. 임베딩 레이어 파라미터: %,d", tokenEmbeddingParameters + positionEmbeddingParameters))

        // --- 2. 트랜스포머 블록 (numberOfLayers 만큼 반복) ---
        var singleBlockParameters = 0L
        val biasParameters = { size: Long -> if (this.bias) size else 0L }

        // a. Multi-Head Self-Attention (MHSA)
        singleBlockParameters += (this.embeddingDimension.toLong() * this.embeddingDimension * 3) + biasParameters(this.embeddingDimension.toLong() * 3)
        singleBlockParameters += (this.embeddingDimension.toLong() * this.embeddingDimension) + biasParameters(this.embeddingDimension.toLong())

        // b. Feed-Forward Network (FFN)
        val feedForwardHiddenSize = this.embeddingDimension * 4
        singleBlockParameters += (this.embeddingDimension.toLong() * feedForwardHiddenSize) + biasParameters(feedForwardHiddenSize.toLong())
        singleBlockParameters += (feedForwardHiddenSize.toLong() * this.embeddingDimension) + biasParameters(this.embeddingDimension.toLong())

        // c. Layer Normalization (블록 당 2개)
        singleBlockParameters += 2 * (this.embeddingDimension.toLong() * 2)

        println(String.format("2. 단일 트랜스포머 블록 파라미터: %,d", singleBlockParameters))
        totalParameters += singleBlockParameters * this.numberOfLayers
        println(String.format("   => 총 트랜스포머 블록 파라미터 (%d개): %,d", this.numberOfLayers, singleBlockParameters * this.numberOfLayers))

        // --- 3. 최종 출력층 ---
        val finalLayerNormParameters = this.embeddingDimension.toLong() * 2
        totalParameters += finalLayerNormParameters
        println(String.format("3. 최종 LayerNorm 파라미터: %,d", finalLayerNormParameters))

        // --- 4. Language Model Head (lmHead) ---
        val languageModelHeadParameters = this.embeddingDimension.toLong() * vocabularySize // bias=false이므로 bias 파라미터 없음
        totalParameters += languageModelHeadParameters
        println(String.format("4. Language Model Head 파라미터: %,d", languageModelHeadParameters))

        println(String.format("총 파라미터 수: %,d", totalParameters))
        return totalParameters
    }

}








