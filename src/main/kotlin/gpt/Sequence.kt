package gpt

import Value

/**
 * 토큰 시퀀스를 나타내는 클래스
 * 
 * 언어 모델에서 토큰들의 임베딩 시퀀스를 표현합니다.
 * [token_count, embedding_dim] 형태의 2차원 배열을 추상화합니다.
 * 
 * @param data 실제 시퀀스 데이터 [token_count, embedding_dim]
 */
class Sequence(data: Array<Array<Value>>) : Matrix(data) {
    
    /** 토큰 개수 */
    val tokenCount: Int get() = rows
    
    /** 임베딩 차원 */
    val embeddingDim: Int get() = cols
    
    
    
    /**
     * 시퀀스의 각 토큰에 함수 적용
     * @param transform 적용할 변환 함수
     * @return 변환된 새로운 시퀀스
     */
    fun mapTokens(transform: (Array<Value>) -> Array<Value>): Sequence {
        return Sequence(mapRows(transform).values)
    }
    
    /**
     * 다른 시퀀스와 요소별 연산
     * @param other 다른 시퀀스
     * @param operation 수행할 연산
     * @return 연산 결과 시퀀스
     */
    fun zipWith(other: Sequence, operation: (Value, Value) -> Value): Sequence {
        return Sequence(super.zipWith(other, operation).values)
    }
    
    companion object {
        /**
         * 빈 시퀀스 생성
         */
        fun empty(): Sequence = Sequence(emptyArray())
        
        /**
         * 원본 배열에서 시퀀스 생성
         */
        fun fromArray(array: Array<Array<Value>>): Sequence = Sequence(array)
    }
}