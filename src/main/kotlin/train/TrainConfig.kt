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
    /** 데이터 파일이 위치한 기본 경로 */
    val dataPath: String = "data",
    /** 훈련된 모델과 체크포인트가 저장될 디렉토리 */
    val modelDir: String = "model",
    /** 전체 훈련 반복 중 검증을 얼마나 자주 실행할지에 대한 비율 (e.g., 0.01 = 1%) */
    val evalIntervalRatio: Float = 0.01f,
    /** 훈련 중 로그를 얼마나 자주 출력할지에 대한 반복 수 간격 */
    val logInterval: Int = 1,
    /** 검증 단계에서 사용할 반복 횟수 */
    val evalIters: Int = 1,
    /** true일 경우, 훈련 없이 검증만 실행 */
    val evalOnly: Boolean = false,
    /** true일 경우, 성능 향상 여부와 관계없이 항상 체크포인트를 저장 */
    val alwaysSaveCheckpoint: Boolean = true,
    /** 모델 초기화 방식 ('scratch': 처음부터 학습, 'resume': 체크포인트에서 이어하기) */
    val initFrom: String = "scratch",
    /** 체크포인트를 저장할 때 사용할 하위 디렉토리 이름 (선택 사항) */
    val subDir: String? = null,

    // 데이터
    /** 사용할 데이터셋의 이름 (e.g., "stories") */
    val dataset: String = "stories",
    /** 그래디언트 누적 단계 수. 실질적인 배치 크기를 늘려 안정적인 학습을 돕습니다. */
    val gradientAccumulationSteps: Int = 4,
    /** 한 번의 반복(iteration)에서 사용할 데이터 샘플의 수 */
    val batchSize: Int = 4,
    /** 모델이 한 번에 처리할 수 있는 최대 토큰 시퀀스 길이 (컨텍스트 윈도우) */
    val blockSize: Int = 24,

    // 모델
    /** 모델의 임베딩 벡터 차원. 모델의 표현력을 결정하는 핵심 하이퍼파라미터. */
    val embeddingDimension: Int = 4,
    /** 모델에 포함된 트랜스포머 블록(레이어)의 수. 모델의 깊이를 결정. */
    val numberOfLayers: Int = 1,
    /** 멀티-헤드 어텐션에서 사용할 헤드의 수. `embeddingDimension`의 약수여야 합니다. */
    val numberOfHeads: Int = 1,
    /** 모델의 선형 레이어에서 편향(bias)을 사용할지 여부 */
    val bias: Boolean = true,
    /** 과적합을 방지하기 위한 드롭아웃 확률 (0.0 ~ 1.0) */
    val dropout: Float = 0.15f,

    // 옵티마이저
    /** 옵티마이저의 학습률. 너무 크면 발산, 너무 작으면 학습이 느려집니다. */
    val learningRate: Float = 5e-4f,
    /** 총 훈련 반복 횟수 */
    val maxIters: Int = 5000,
    /** AdamW 옵티마이저의 가중치 감쇠(weight decay) 계수. L2 정규화와 유사한 효과. */
    val weightDecay: Float = 0.05f,
    /** Adam 옵티마이저의 1차 모멘트 추정(momentum)을 위한 지수 감쇠율 */
    val beta1: Float = 0.9f,
    /** Adam 옵티마이저의 2차 모멘트 추정(RMSProp)을 위한 지수 감쇠율 */
    val beta2: Float = 0.99f,
    /** 그래디언트 폭발을 방지하기 위한 그래디언트 클리핑(clipping) 임계값 */
    val gradClip: Float = 1.0f,

    // 학습률 스케줄
    /** 학습률을 스케줄에 따라 감소시킬지 여부 */
    val decayLr: Boolean = true,
    /** 전체 훈련 반복 중 학습률을 점진적으로 증가시키는 '웜업' 기간의 비율 */
    val warmupRatio: Float = 0.01f,
    /** 전체 훈련 반복 중 학습률이 감소하는 기간의 비율 */
    val learningRateDecayRatio: Float = 0.8f,
    /** 학습률 스케줄러가 도달할 수 있는 최소 학습률 */
    val minimumLearningRate: Float = 1e-5f
) {
    // 계산된 속성들
    /** 계산된 속성: 웜업 반복 횟수 */
    val warmupIters: Int get() = (maxIters * warmupRatio).toInt()
    /** 계산된 속성: 학습률 감소 반복 횟수 */
    val learningRateDecayIterations: Int get() = (maxIters * learningRateDecayRatio).toInt()
    /** 계산된 속성: 검증 간격 (반복 횟수) */
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








