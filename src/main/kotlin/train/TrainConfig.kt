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
    val gradientAccumulationSteps: Int = 1, // 4 -> 1로 줄여서 속도 향상
    val batchSize: Int = 8, // 16 -> 8로 줄여서 메모리/속도 최적화
    val blockSize: Int = 32, // 64 -> 32로 줄여서 시퀀스 길이 최소화

    // 모델
    val nLayer: Int = 3, // 레이어 수 최소화
    val nHead: Int = 5, // 헤드 수 최소화
    val nEmbd: Int = 32, // 임베딩 차원 최소화
    val bias: Boolean = true,
    val dropout: Float = 0.1f, // Dropout 확률

    // 옵티마이저
    val learningRate: Float = 3e-3f,
    val maxIters: Int = 5000,
    val weightDecay: Float = 1e-2f,
    val beta1: Float = 0.9f,
    val beta2: Float = 0.99f,
    val gradClip: Float = 1.0f,

    // 학습률 스케줄
    val decayLr: Boolean = true,
    val warmupRatio: Float = 0.2f, // maxIters의 10%를 warmup으로 사용
    val lrDecayRatio: Float = 0.8f, // maxIters의 100%까지 decay (훈련 끝까지)
    val minLr: Float = 1e-4f
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








