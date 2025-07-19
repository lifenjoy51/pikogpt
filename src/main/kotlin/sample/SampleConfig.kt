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
    val samplingTemperature: Float = 0.8f,

    /** Top-K 샘플링 값 (가장 가능성 높은 K개 토큰만 고려, 0이면 비활성화) */
    val topKFilteringSize: Int = 100,

    /** 랜덤 시드 값 (재현 가능한 결과를 위해 사용) */
    val randomSeed: Int = 1337
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
