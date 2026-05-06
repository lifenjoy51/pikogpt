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
    /** 훈련 중 로그 출력 빈도 */
    val logInterval: Int = 1,
    /** 검증 단계에서 사용할 반복 횟수 */
    val evalIters: Int = 10,
    /** true일 경우, 훈련 없이 검증만 실행 */
    val evalOnly: Boolean = false,
    /** true일 경우, 성능 향상 여부와 관계없이 항상 체크포인트를 저장 */
    val alwaysSaveCheckpoint: Boolean = true,
    /** 모델 초기화 방식 ('scratch': 처음부터 학습, 'resume': 체크포인트에서 이어하기, 'pretrain_weights': pretrain 가중치만 로드 + optimizer reset). */
    val initFrom: String = "scratch",
    /** 체크포인트를 저장할 때 사용할 하위 디렉토리 이름 (선택 사항) */
    val modelCheckpointDir: String? = null,
    /** initFrom='pretrain_weights' 일 때 가중치를 로드할 ckpt 디렉터리 경로. */
    val pretrainCheckpointDir: String? = null,
    /** Replay용 보조 train.bin 경로. null이면 단일 데이터로더 사용. IT(finetune) 단계에서 BASE 데이터를 일정 비율 섞어 catastrophic forgetting을 줄일 때 지정. */
    val replayDataPath: String? = null,
    /** Replay 비율 (0.0~1.0). 미니배치 시퀀스마다 Bernoulli(p=replayRatio)로 replay 데이터에서 추출. 0이면 비활성. */
    val replayRatio: Float = 0.0f,
    /** **두 번째** replay용 train.bin 경로 (multi-replay, 예: 3-stage curriculum 마지막 단계에서 dict + wiki 별도 path). null이면 비활성. */
    val replayDataPath2: String? = null,
    /** 두 번째 replay 비율. `replayRatio + replayRatio2` 합이 1.0 이하여야 함. 0이면 비활성. */
    val replayRatio2: Float = 0.0f,
    /** Early stop patience: best loss 갱신 없이 N번 연속 eval되면 학습 조기 종료. 0이면 비활성(=maxIters까지). */
    val earlyStopPatience: Int = 0,
    /** 체크포인트 저장 직후 자동 샘플링에 사용할 프롬프트 목록. null이면 trainer의 기본 prompt 사용. */
    val samplePrompts: List<String>? = null,
    /** true면 RecordAwareDataLoader 사용 — 매 시퀀스가 한 record(=한 <|bos|>...<|eos|>) 안에 머무는 두 단계 sampling. */
    val recordAwareSampling: Boolean = false,
    /** true면 ChunkAnchoredDataLoader 사용 — record 안 stride=blockSize chunk anchor + jitter sampling.
     * `recordAwareSampling`과 동시에 true면 이 옵션이 우선. binding 학습 신호 강화 (anchor당 학습 빈도 ~26×). */
    val chunkAnchoredSampling: Boolean = false,

    // 데이터
    /** 사용할 데이터셋의 이름 (e.g., "base", "it") */
    val dataset: String = "default",
    /** 그래디언트 누적 단계 수. 실질적인 배치 크기를 늘려 안정적인 학습을 돕습니다. [메모리: 낮음, 시간: 선형 증가] */
    val gradientAccumulationSteps: Int = 2,
    /** 한 번의 반복(iteration)에서 사용할 데이터 샘플의 수 [메모리: 선형 증가, 시간: 선형 증가] */
    val batchSize: Int = 8,
    /** 모델이 한 번에 처리할 수 있는 최대 토큰 시퀀스 길이 (컨텍스트 윈도우)
     *  더 긴 컨텍스트 학습: 이야기 전체를 한 번에 처리
     *  일관성 향상: 문장 간 연결성과 스토리 흐름 학습
     *  [메모리: 제곱 증가(어텐션), 시간: 제곱 증가] */
    val blockSize: Int = 48,

    // 모델
    /** 모델의 임베딩 벡터 차원. 모델의 표현력을 결정하는 핵심 하이퍼파라미터.
     *  표현력 향상: 더 풍부한 의미 표현
     *  복잡한 패턴 학습: 언어의 미묘한 뉘앙스 캡처
     *  [메모리: 제곱 증가, 시간: 제곱 증가] */
    val embeddingDimension: Int = 16,
    /** 모델에 포함된 트랜스포머 블록(레이어)의 수. 모델의 깊이를 결정. [메모리: 선형 증가, 시간: 선형 증가] */
    val numberOfLayers: Int = 2,
    /** 멀티-헤드 어텐션에서 사용할 헤드의 수. `embeddingDimension`의 약수여야 합니다. [메모리: 낮음, 시간: 거의 변화 없음] */
    val numberOfHeads: Int = 2,
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
    /** Label smoothing 계수(0.0~1.0). 0이면 비활성, 0.1이면 target 분포를 (1-0.1)·onehot + 0.1·uniform로 대체. Overconfidence 완화. 현재는 `vec` 백엔드만 사용. */
    val labelSmoothing: Float = 0.0f,
    /** Weight tying — token_embedding과 lm_head 공유 (vec 백엔드만 적용). vocab×dim 절약 + 학습 신호 강화. 기본 true. */
    val tieWeights: Boolean = true,
    /** MLP activation — `"gelu"`(default, GPT-2 스타일) 또는 `"swiglu"`(Llama 스타일, hidden=8/3·dim). */
    val mlpActivation: String = "gelu",
    /** Position encoding — `"learned"`(default, GPT-2 스타일) 또는 `"rope"`(Q·K 회전 주입, position param 제거). */
    val positionEncoding: String = "learned",

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
     * @param vocabularySize 모델의 어휘 사전 크기
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

        // a. Multi-Head Self-Attention (MHSA) - 4개의 별도 Linear 레이어 (Q, K, V, Output)
        singleBlockParameters += 4 * ((this.embeddingDimension.toLong() * this.embeddingDimension) + biasParameters(this.embeddingDimension.toLong()))

        // b. Feed-Forward Network (FFN)
        val feedForwardHiddenSize = this.embeddingDimension * 4
        singleBlockParameters += (this.embeddingDimension.toLong() * feedForwardHiddenSize) + biasParameters(feedForwardHiddenSize.toLong())
        singleBlockParameters += (feedForwardHiddenSize.toLong() * this.embeddingDimension) + biasParameters(this.embeddingDimension.toLong())

        // c. Layer Normalization (블록 당 2개) - 각각 scale + shift 파라미터
        val layerNormParams = if (this.bias) 2 * this.embeddingDimension.toLong() else this.embeddingDimension.toLong()
        singleBlockParameters += 2 * layerNormParams

        println(String.format("2. 단일 트랜스포머 블록 파라미터: %,d", singleBlockParameters))
        totalParameters += singleBlockParameters * this.numberOfLayers
        println(String.format("   => 총 트랜스포머 블록 파라미터 (%d개): %,d", this.numberOfLayers, singleBlockParameters * this.numberOfLayers))

        // --- 3. 최종 출력층 ---
        val finalLayerNormParameters = if (this.bias) 2 * this.embeddingDimension.toLong() else this.embeddingDimension.toLong()
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








