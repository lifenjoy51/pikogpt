package gpt

import Value

/**
 * Value 타입의 2차원 행렬을 나타내는 기본 클래스
 * 
 * 일반적인 행렬 연산을 위한 래퍼 클래스입니다.
 * 드롭아웃, 어텐션 스코어 등 다양한 2차원 연산에서 사용됩니다.
 * EmbeddingTable, Sequence, Logits 등이 이 클래스를 상속합니다.
 * 
 * @param data 실제 행렬 데이터 [rows, cols]
 */
open class Matrix(protected val data: Array<Array<Value>>) {
    
    /** 행의 수 */
    val rows: Int get() = data.size
    
    /** 열의 수 */
    val cols: Int get() = if (data.isNotEmpty()) data[0].size else 0
    
    /** 원본 데이터 접근 */
    val values: Array<Array<Value>> get() = data
    
    /**
     * 특정 행 가져오기
     * @param row 행 인덱스 (0부터 시작)
     * @return 해당 행의 데이터
     */
    open operator fun get(row: Int): Array<Value> = data[row]
    
    /**
     * 특정 원소 가져오기
     * @param row 행 인덱스
     * @param col 열 인덱스  
     * @return 해당 위치의 값
     */
    operator fun get(row: Int, col: Int): Value = data[row][col]
    
    /**
     * 특정 원소 설정하기
     * @param row 행 인덱스
     * @param col 열 인덱스
     * @param value 설정할 값
     */
    open operator fun set(row: Int, col: Int, value: Value) {
        data[row][col] = value
    }
    
    /**
     * 행렬의 각 원소에 함수 적용
     * @param transform 적용할 변환 함수
     * @return 변환된 새로운 행렬
     */
    fun map(transform: (Value) -> Value): Matrix {
        val result = Array(rows) { i ->
            Array(cols) { j ->
                transform(data[i][j])
            }
        }
        return Matrix(result)
    }
    
    /**
     * 행렬의 각 행에 함수 적용
     * @param transform 적용할 변환 함수
     * @return 변환된 새로운 행렬
     */
    fun mapRows(transform: (Array<Value>) -> Array<Value>): Matrix {
        return Matrix(data.map(transform).toTypedArray())
    }
    
    /**
     * 다른 행렬과 원소별 연산
     * @param other 다른 행렬
     * @param operation 수행할 연산
     * @return 연산 결과 행렬
     */
    fun zipWith(other: Matrix, operation: (Value, Value) -> Value): Matrix {
        require(rows == other.rows && cols == other.cols) {
            "행렬 차원이 일치하지 않습니다: ($rows, $cols) vs (${other.rows}, ${other.cols})"
        }
        
        val result = Array(rows) { i ->
            Array(cols) { j ->
                operation(data[i][j], other.data[i][j])
            }
        }
        return Matrix(result)
    }
    
    /**
     * 모든 학습 가능한 파라미터 반환
     * @return 모든 파라미터의 리스트
     */
    open fun parameters(): List<Value> {
        return data.flatMap { it.toList() }
    }
    
    /**
     * 시퀀스로 변환 (시퀀스 관점으로 해석할 때 사용)
     * @return 시퀀스 객체
     */
    fun toSequence(): Sequence = Sequence(data)
    
    /**
     * 로짓으로 변환 (로짓 관점으로 해석할 때 사용)
     * @return 로짓 객체
     */
    fun toLogits(): Logits = Logits(data)
    
    /**
     * 임베딩 테이블로 변환 (임베딩 테이블 관점으로 해석할 때 사용)
     * @return 임베딩 테이블 객체
     */
    fun toEmbeddingTable(): EmbeddingTable = EmbeddingTable.fromArray(data)
    
    companion object {
        /**
         * 원본 배열에서 행렬 생성
         */
        fun fromArray(array: Array<Array<Value>>): Matrix = Matrix(array)
        
        /**
         * 시퀀스에서 행렬 생성
         */
        fun fromSequence(sequence: Sequence): Matrix = Matrix(sequence.values)
        
        /**
         * 로짓에서 행렬 생성
         */
        fun fromLogits(logits: Logits): Matrix = Matrix(logits.values)
        
        /**
         * 영행렬 생성
         * @param rows 행의 수
         * @param cols 열의 수
         * @return 모든 원소가 0인 행렬
         */
        fun zeros(rows: Int, cols: Int): Matrix {
            val data = Array(rows) { Array(cols) { Value(0.0f) } }
            return Matrix(data)
        }
    }
}