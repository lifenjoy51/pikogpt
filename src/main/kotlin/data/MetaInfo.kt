package data

import kotlinx.serialization.Serializable

/**
 * 어휘 사전 메타데이터.
 *
 * 모델과 함께 저장되어 Sampler가 **학습 시와 동일한 토큰화**를 재생할 수 있게 한다.
 * - 어휘 (vocabSize + stringToIndex + indexToString): 기본 매핑
 * - merges: BPE 병합 규칙. 순서대로 적용.
 * - lowercase / useWordPreTokenize: 학습 시 사용한 전처리 플래그.
 *
 * 새 필드는 기본값이 있어 구버전 체크포인트 (merges 없는 meta.json)도 그대로 파싱된다.
 * 그 경우 Sampler는 그리디 longest-match로 폴백한다.
 */
@Serializable
data class MetaInfo(
    /** 전체 어휘 사전의 크기 (가능한 모든 토큰의 수) */
    val vocabularySize: Int,

    /** 토큰 ID에서 문자열로의 매핑 (인덱스 → 토큰) */
    val indexToString: Map<Int, String>,

    /** 문자열에서 토큰 ID로의 매핑 (토큰 → 인덱스) */
    val stringToIndex: Map<String, Int>,

    /**
     * BPE 병합 규칙 (new; 없으면 빈 리스트).
     * 각 원소는 `[first, second]`로 직렬화되며 학습된 순서 그대로 유지된다.
     */
    val merges: List<List<String>> = emptyList(),

    /** 학습 시 소문자 정규화를 적용했는지. Sampler가 동일한 전처리를 재생하려면 필요. */
    val lowercase: Boolean = false,

    /** 학습 시 단어 pre-tokenize(GPT-2 스타일)를 사용했는지. */
    val useWordPreTokenize: Boolean = false,

    /** BPE 학습 시 사용된 특수 토큰 목록 (학습 순서 = ID 순서). 기본 배치: eos=0, unk=1, bos=2. */
    val specialTokens: List<String> = listOf("<|eos|>", "<|unk|>", "<|bos|>"),
) {
    // 호환성을 위한 별칭 속성들
    val vocabSize: Int get() = vocabularySize
    val itos: Map<Int, String> get() = indexToString
    val stoi: Map<String, Int> get() = stringToIndex
}
