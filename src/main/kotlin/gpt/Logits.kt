package gpt

import Value

/**
 * 모델의 로짓 출력을 나타내는 클래스
 * 
 * 언어 모델에서 각 위치별 어휘 확률 분포를 표현합니다.
 * [token_count, vocab_size] 형태의 2차원 배열을 추상화합니다.
 * 
 * @param data 실제 로짓 데이터 [token_count, vocab_size]
 */
class Logits(data: Array<Array<Value>>) : Matrix(data) {
    
    /** 토큰 개수 */
    val tokenCount: Int get() = rows
    
    /** 어휘 크기 */
    val vocabSize: Int get() = cols
    
    
    
    /**
     * 특정 위치의 특정 토큰 로짓 값 가져오기
     * @param position 시퀀스 내 위치
     * @param tokenId 토큰 ID
     * @return 해당 토큰의 로짓 값
     */
    fun getLogit(position: Int, tokenId: Int): Value = get(position, tokenId)
    
    /**
     * 마지막 위치의 로짓만 가져오기 (다음 토큰 예측용)
     * @return 마지막 위치의 어휘 로짓 분포
     */
    fun getLastPositionLogits(): Array<Value> = get(tokenCount - 1)
    
    /**
     * 각 위치에서 가장 높은 확률을 가진 토큰 ID 찾기
     * @return 각 위치별 최고 확률 토큰 ID 배열
     */
    fun getArgMax(): IntArray {
        return (0 until tokenCount).map { position ->
            val logitRow = get(position)
            logitRow.indices.maxByOrNull { logitRow[it].scalarValue } ?: 0
        }.toIntArray()
    }
    
    /**
     * 소프트맥스 적용하여 확률 분포로 변환
     * @return 확률 분포로 변환된 로짓
     */
    fun softmax(): Logits {
        val probabilities = mapRows { logitRow ->
            // 수치 안정성을 위해 최댓값을 빼줌
            val maxLogit = logitRow.maxByOrNull { it.scalarValue } ?: Value(0.0f)
            val expValues = logitRow.map { (it - maxLogit).exp() }.toTypedArray()
            val sumExp = expValues.reduce { acc, exp -> acc + exp }
            expValues.map { it / sumExp }.toTypedArray()
        }
        
        return Logits(probabilities.values)
    }
    
    companion object {
        /**
         * 원본 배열에서 로짓 생성
         */
        fun fromArray(array: Array<Array<Value>>): Logits = Logits(array)
    }
}