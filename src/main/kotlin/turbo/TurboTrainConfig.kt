package turbo

import kotlinx.serialization.Serializable

/**
 * Turbo 백엔드 전용 학습 설정.
 *
 * Scalar 백엔드의 [train.TrainConfig]와 같은 기본 필드(데이터 경로/모델 차원/옵티마이저 등)를
 * 그대로 가진다. 추가로 turbo 백엔드 전용 옵션(replay, record-aware/chunk-anchored sampling,
 * label smoothing, weight tying, SwiGLU/RoPE 등)을 포함한다.
 *
 * 두 백엔드의 의도를 코드에서 분명히 분리하기 위해 별도 클래스로 둔다.
 * 기본 필드만 공유하고 싶을 때는 `toScalarTrainConfig()`로 변환할 수 있다.
 */
@Serializable
data class TurboTrainConfig(
    // ============================================================================
    // 기본 (Scalar와 동일)
    // ============================================================================

    // I/O
    val dataPath: String = "data",
    val modelDir: String = "model",
    /** 실험 이름. 같은 datasetName을 공유하는 진입점들을 분리하기 위한 사람이 정한 식별자.
     *  체크포인트 경로: `${modelDir}/${datasetName}/${expName}/v0001/`. */
    val expName: String = "main",
    val evalIntervalRatio: Float = 0.01f,
    val logInterval: Int = 1,
    val evalIters: Int = 10,
    val evalOnly: Boolean = false,
    val alwaysSaveCheckpoint: Boolean = true,
    /** 모델 초기화 방식 ('scratch': 처음부터 학습, 'resume': 체크포인트에서 이어하기, 'pretrain_weights': pretrain 가중치만 로드 + optimizer reset). */
    val initFrom: String = "scratch",
    val modelCheckpointDir: String? = null,

    // 데이터
    val gradientAccumulationSteps: Int = 2,
    val batchSize: Int = 8,
    val blockSize: Int = 48,

    // 모델
    val embeddingDimension: Int = 16,
    val numberOfLayers: Int = 2,
    val numberOfHeads: Int = 2,
    val bias: Boolean = true,
    val dropout: Float = 0.15f,

    // 옵티마이저
    val learningRate: Float = 5e-4f,
    val maxIters: Int = 5000,
    val weightDecay: Float = 0.05f,
    val beta1: Float = 0.9f,
    val beta2: Float = 0.99f,
    val gradClip: Float = 1.0f,

    // 학습률 스케줄
    val decayLr: Boolean = true,
    val warmupRatio: Float = 0.01f,
    val learningRateDecayRatio: Float = 0.8f,
    val minimumLearningRate: Float = 1e-5f,

    // ============================================================================
    // Turbo 전용
    // ============================================================================

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
    /** 데이터셋 식별자 (예: "base", "it"). */
    val dataset: String = "default",
    /** Label smoothing 계수(0.0~1.0). 0이면 비활성, 0.1이면 target 분포를 (1-0.1)·onehot + 0.1·uniform로 대체. */
    val labelSmoothing: Float = 0.0f,
    /** Weight tying — token_embedding과 lm_head 공유. vocab×dim 절약 + 학습 신호 강화. */
    val tieWeights: Boolean = true,
    /** MLP activation — `"gelu"`(default, GPT-2 스타일) 또는 `"swiglu"`(Llama 스타일, hidden=8/3·dim). */
    val mlpActivation: String = "gelu",
    /** Position encoding — `"learned"`(default, GPT-2 스타일) 또는 `"rope"`(Q·K 회전 주입, position param 제거). */
    val positionEncoding: String = "learned",
    /** Lemma stream을 분리해 가중치 낮춰 sampling. null이면 단일 train.bin 사용. 값(0~1)이 secondaryProb.
     *  dataPath/train_lemma.bin(secondary) + dataPath/train_other.bin(primary) 둘 다 필수. */
    val lemmaSamplingRatio: Float? = null,
) {
    // 계산된 속성들
    val warmupIters: Int get() = (maxIters * warmupRatio).toInt()
    val learningRateDecayIterations: Int get() = (maxIters * learningRateDecayRatio).toInt()
    val evalInterval: Int get() = (maxIters * evalIntervalRatio).toInt()
}
