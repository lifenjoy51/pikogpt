package sample

import kotlinx.serialization.Serializable

/**
 * 텍스트 생성 샘플링 설정
 *
 * GPT 모델에서 텍스트를 생성할 때 사용되는 모든 하이퍼파라미터를 정의합니다.
 * 이 설정들은 생성되는 텍스트의 다양성, 창의성, 품질을 제어합니다.
 */
@Serializable
data class SampleConfig(
    /** 모델 초기화 방식 ('resume': 체크포인트에서 로드, 'scratch': 랜덤 초기화) */
    val modelInitializationMode: String = "resume",

    /** 학습된 모델이 저장된 디렉토리 경로 */
    val modelDirectoryPath: String = "model",

    /** 동일한 프롬프트에 대해 생성할 결과의 수 (다양성 확보용) */
    val numberOfSamples: Int = 10,

    /** 생성할 최대 새 토큰 수 (생성 길이 제어) */
    val maximumNewTokens: Int = 50,

    /** 샘플링 온도 (0.0: 결정론적, 1.0: 창의적, >1.0: 매우 창의적) */
    val samplingTemperature: Float = 1.0f,

    /** Top-K 샘플링 값 (가장 가능성 높은 K개 토큰만 고려, 0이면 비활성화) */
    val topKFilteringSize: Int = 100,

    /** 랜덤 시드 값 (재현 가능한 결과를 위해 사용) */
    val randomSeed: Int = 51,

    /** 생성을 중단할 토큰 id 목록 (기본: EOS=0만). 대화 데이터에서는 `<|turn|>` id도 추가하면
     *  single-turn 응답을 받을 수 있다. SamplePromptsFromFile이 meta.json을 보고 자동 설정. */
    val stopTokenIds: List<Int> = listOf(0),

    /**
     * Top-p (nucleus) 샘플링 임계값 (0.0 < p ≤ 1.0). 누적 확률이 p에 도달할 때까지의 토큰만
     * 후보로 남김. 1.0이면 비활성. top-k와 병행 시 둘 다 적용 (top-k 먼저 → 그 안에서 top-p).
     * 권장값: 0.9-0.95 (보수적), 0.85-0.9 (다양성 강조).
     */
    val topProbabilityThreshold: Float = 1.0f,

    /**
     * Repetition penalty (1.0 = 비활성). 직전 [repetitionWindow] 토큰에 등장한 토큰의 logit을
     *   logit / penalty (logit > 0)
     *   logit * penalty (logit < 0)
     * 로 보정해 반복 확률 낮춤. 표준 hugging face 공식. 권장값 1.1-1.3.
     */
    val repetitionPenalty: Float = 1.0f,

    /** Repetition penalty 적용 시 거슬러 보는 최근 토큰 수. */
    val repetitionWindow: Int = 64
) {
    // 호환성을 위한 별칭 속성들
    val initFrom: String get() = modelInitializationMode
    val modelDir: String get() = modelDirectoryPath
    val numSamples: Int get() = numberOfSamples
    val maxNewTokens: Int get() = maximumNewTokens
    val temperature: Float get() = samplingTemperature
    val topK: Int get() = topKFilteringSize
    val seed: Int get() = randomSeed
}
