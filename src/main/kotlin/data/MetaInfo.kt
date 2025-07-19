package data

import kotlinx.serialization.Serializable

/**
 * 메타 정보를 저장할 데이터 클래스
 */
@Serializable
data class MetaInfo(
    val vocabSize: Int,
    val itos: Map<Int, String>,
    val stoi: Map<String, Int>
)