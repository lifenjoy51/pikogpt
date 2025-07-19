package train

import kotlinx.serialization.Serializable

// 훈련 설정
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
    val nEmbd: Int = 4, // 가장 기본적인 성능 향상 방법입니다. 모델이 각 토큰을 표현하는 정보의 양을 늘려줍니다.
    val nLayer: Int = 1, // 모델의 깊이를 늘려 더 복잡하고 추상적인 패턴을 학습하게 합니다.
    val nHead: Int = 1, // 어텐션 메커니즘이 한 번에 다양한 종류의 관계를 보도록 합니다. nHead는 nEmbd의 약수여야 합니다.
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
    val lrDecayRatio: Float = 0.8f, // maxIters의 100%까지 decay (훈련 끝까지)
    val minLr: Float = 1e-5f
) {
    // 계산된 속성들
    val warmupIters: Int get() = (maxIters * warmupRatio).toInt()
    val lrDecayIters: Int get() = (maxIters * lrDecayRatio).toInt()
    val evalInterval: Int get() = (maxIters * evalIntervalRatio).toInt()

    /**
     * TrainConfig의 확장 함수로, 총 파라미터 수를 계산합니다.
     * @param vocabSize 모델의 어휘 사전 크기 (e.g., 65 for shakespeare_char)
     * @return Long 타입의 총 파라미터 수
     */
    fun calculateTotalParameters(vocabSize: Int): Long {
        var totalParams = 0L

        // --- 1. 임베딩 레이어 ---
        val tokenEmbeddingParams = vocabSize.toLong() * this.nEmbd
        val positionEmbeddingParams = this.blockSize.toLong() * this.nEmbd
        totalParams += tokenEmbeddingParams + positionEmbeddingParams
        println(String.format("1. 임베딩 레이어 파라미터: %,d", tokenEmbeddingParams + positionEmbeddingParams))

        // --- 2. 트랜스포머 블록 (nLayer 만큼 반복) ---
        var singleBlockParams = 0L
        val biasParam = { size: Long -> if (this.bias) size else 0L }

        // a. Multi-Head Self-Attention (MHSA)
        singleBlockParams += (this.nEmbd.toLong() * this.nEmbd * 3) + biasParam(this.nEmbd.toLong() * 3)
        singleBlockParams += (this.nEmbd.toLong() * this.nEmbd) + biasParam(this.nEmbd.toLong())

        // b. Feed-Forward Network (FFN)
        val ffnHiddenSize = this.nEmbd * 4
        singleBlockParams += (this.nEmbd.toLong() * ffnHiddenSize) + biasParam(ffnHiddenSize.toLong())
        singleBlockParams += (ffnHiddenSize.toLong() * this.nEmbd) + biasParam(this.nEmbd.toLong())

        // c. Layer Normalization (블록 당 2개)
        singleBlockParams += 2 * (this.nEmbd.toLong() * 2)

        println(String.format("2. 단일 트랜스포머 블록 파라미터: %,d", singleBlockParams))
        totalParams += singleBlockParams * this.nLayer
        println(String.format("   => 총 트랜스포머 블록 파라미터 (%d개): %,d", this.nLayer, singleBlockParams * this.nLayer))

        // --- 3. 최종 출력층 ---
        val finalLayerNormParams = this.nEmbd.toLong() * 2
        totalParams += finalLayerNormParams
        println(String.format("3. 최종 LayerNorm 파라미터: %,d", finalLayerNormParams))

        // --- 4. Language Model Head (lmHead) ---
        val lmHeadParams = this.nEmbd.toLong() * vocabSize // bias=false이므로 bias 파라미터 없음
        totalParams += lmHeadParams
        println(String.format("4. Language Model Head 파라미터: %,d", lmHeadParams))

        return totalParams
    }

}








