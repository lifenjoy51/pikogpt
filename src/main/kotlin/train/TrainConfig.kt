package train

import kotlinx.serialization.Serializable

/**
 * Scalar 백엔드 학습 설정.
 *
 * 모델 아키텍처(layers/heads/embd), 학습 하이퍼파라미터(lr/maxIters/grad clipping),
 * 학습률 스케줄(warmup + cosine decay), 체크포인트/평가 정책을 한 곳에 모은다.
 *
 * Turbo 전용 옵션(replay, record-aware sampling, swiglu/rope, label smoothing 등)은
 * `turbo.TurboTrainConfig`로 분리되어 있다.
 */
@Serializable
data class TrainConfig(
    // I/O
    /** 데이터 파일이 위치한 기본 경로 */
    val dataPath: String = "data",
    /** 훈련된 모델과 체크포인트가 저장될 디렉토리 */
    val modelDir: String = "model",
    /** 실험 이름. 같은 datasetName을 공유하는 진입점들을 분리하기 위한 사람이 정한 식별자.
     *  체크포인트 경로: `${modelDir}/${datasetName}/${expName}/v0001/`. */
    val expName: String = "main",
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
    /** 모델 초기화 방식 ('scratch': 처음부터 학습, 'resume': 체크포인트에서 이어하기). */
    val initFrom: String = "scratch",
    /** 체크포인트를 저장할 때 사용할 하위 디렉토리 이름 (선택 사항) */
    val modelCheckpointDir: String? = null,

    // 데이터
    /** 그래디언트 누적 단계 수. 실질적인 배치 크기를 늘려 안정적인 학습을 돕습니다. [메모리: 낮음, 시간: 선형 증가] */
    val gradientAccumulationSteps: Int = 2,
    /** 한 번의 반복(iteration)에서 사용할 데이터 샘플의 수 [메모리: 선형 증가, 시간: 선형 증가] */
    val batchSize: Int = 8,
    /** 모델이 한 번에 처리할 수 있는 최대 토큰 시퀀스 길이 (컨텍스트 윈도우)
     *  [메모리: 제곱 증가(어텐션), 시간: 제곱 증가] */
    val blockSize: Int = 48,

    // 모델
    /** 모델의 임베딩 벡터 차원. 모델의 표현력을 결정하는 핵심 하이퍼파라미터.
     *  [메모리: 제곱 증가, 시간: 제곱 증가] */
    val embeddingDimension: Int = 16,
    /** 모델에 포함된 트랜스포머 블록(레이어)의 수. [메모리: 선형 증가, 시간: 선형 증가] */
    val numberOfLayers: Int = 2,
    /** 멀티-헤드 어텐션에서 사용할 헤드의 수. `embeddingDimension`의 약수여야 합니다. */
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

    // 학습률 스케줄
    /** 학습률을 스케줄에 따라 감소시킬지 여부 */
    val decayLr: Boolean = true,
    /** 전체 훈련 반복 중 학습률을 점진적으로 증가시키는 '웜업' 기간의 비율 */
    val warmupRatio: Float = 0.01f,
    /** 전체 훈련 반복 중 학습률이 감소하는 기간의 비율 */
    val learningRateDecayRatio: Float = 0.8f,
    /** 학습률 스케줄러가 도달할 수 있는 최소 학습률 */
    val minimumLearningRate: Float = 1e-5f,
) {
    // 계산된 속성들
    /** 계산된 속성: 웜업 반복 횟수 */
    val warmupIters: Int get() = (maxIters * warmupRatio).toInt()
    /** 계산된 속성: 학습률 감소 반복 횟수 */
    val learningRateDecayIterations: Int get() = (maxIters * learningRateDecayRatio).toInt()
    /** 계산된 속성: 검증 간격 (반복 횟수) */
    val evalInterval: Int get() = (maxIters * evalIntervalRatio).toInt()
}
