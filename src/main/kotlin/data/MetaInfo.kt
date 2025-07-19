package data

import kotlinx.serialization.Serializable

/**
 * 어휘 사전 메타데이터 클래스
 *
 * 모델이 사용하는 어휘 사전의 모든 매핑 정보를 저장합니다.
 * 텍스트와 토큰 ID 간의 양방향 변환을 위한 필수 정보를 포함합니다.
 *
 * JSON 직렬화가 가능하도록 설계되어 체크포인트와 함께 저장됩니다.
 */
@Serializable
data class MetaInfo(
    /** 전체 어휘 사전의 크기 (가능한 모든 토큰의 수) */
    val vocabularySize: Int,

    /** 토큰 ID에서 문자열로의 매핑 (인덱스 → 토큰) */
    val indexToString: Map<Int, String>,

    /** 문자열에서 토큰 ID로의 매핑 (토큰 → 인덱스) */
    val stringToIndex: Map<String, Int>
) {
    // 호환성을 위한 별칭 속성들
    val vocabSize: Int get() = vocabularySize
    val itos: Map<Int, String> get() = indexToString
    val stoi: Map<String, Int> get() = stringToIndex
}