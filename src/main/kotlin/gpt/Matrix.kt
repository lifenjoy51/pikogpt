package gpt

import Value

/**
 * Value 2D 행렬. 스칼라 백엔드의 유일한 텐서 표현.
 *
 * 의미적 해석은 사용처에 따라 다릅니다:
 *   - 토큰 시퀀스: rows = 토큰 위치, cols = 임베딩 차원
 *   - 어텐션 스코어: rows = query 위치, cols = key 위치
 *   - 로짓: rows = 토큰 위치, cols = vocab 크기
 *   - 임베딩 테이블: rows = vocab/위치, cols = 임베딩 차원
 *
 * 추상화는 일부러 얇습니다. 행/열 인덱싱 + `mapRows` + `zipWith` + 의미적 helper(`lastRow`,
 * `softmaxRows`, `argMaxRows`). 학습자가 필요하면 raw `Array<Array<Value>>`로도 쉽게 내려갈
 * 수 있도록.
 */
open class Matrix(protected val data: Array<Array<Value>>) {

    /** 행 개수. */
    val rows: Int get() = data.size

    /** 열 개수. */
    val cols: Int get() = if (data.isNotEmpty()) data[0].size else 0

    /** 원본 데이터 직접 접근 (parameters() 등에서 사용). */
    val values: Array<Array<Value>> get() = data

    open operator fun get(row: Int): Array<Value> = data[row]

    operator fun get(row: Int, col: Int): Value = data[row][col]

    open operator fun set(row: Int, col: Int, value: Value) {
        data[row][col] = value
    }

    /**
     * 각 행에 변환을 적용해 새 행렬을 반환.
     *
     * 사용처: LayerNorm을 토큰별로 적용, MLP를 토큰별로 적용 등.
     */
    fun mapRows(transform: (Array<Value>) -> Array<Value>): Matrix {
        return Matrix(data.map(transform).toTypedArray())
    }

    /**
     * 같은 형태의 두 행렬을 원소별 결합. 잔여 연결(`x + y`) 계산에 자주 사용.
     */
    fun zipWith(other: Matrix, operation: (Value, Value) -> Value): Matrix {
        require(rows == other.rows && cols == other.cols) {
            "행렬 차원 불일치: ($rows, $cols) vs (${other.rows}, ${other.cols})"
        }
        val result = Array(rows) { i ->
            Array(cols) { j ->
                operation(data[i][j], other.data[i][j])
            }
        }
        return Matrix(result)
    }

    /** 모든 원소를 학습 가능한 파라미터로 노출 (옵티마이저용). */
    open fun parameters(): List<Value> = data.flatMap { it.toList() }

    /**
     * 마지막 행. 자기회귀 언어 모델에서 "다음 토큰을 예측할 위치의 logits"를 가져올 때 사용.
     */
    fun lastRow(): Array<Value> = data[rows - 1]

    /**
     * 행별 softmax — logits를 확률 분포로 변환.
     *
     * 수치 안정화를 위해 각 행에서 최대값을 빼고 exp를 계산합니다 (max-trick).
     */
    fun softmaxRows(): Matrix = mapRows { row ->
        val maxLogit = row.maxByOrNull { it.scalarValue } ?: Value.ZERO
        val expValues = row.map { (it - maxLogit).exp() }.toTypedArray()
        val sumExp = expValues.reduce { acc, e -> acc + e }
        expValues.map { it / sumExp }.toTypedArray()
    }

    /** 각 행의 argmax 인덱스. greedy 디코딩이나 단순 평가용. */
    fun argMaxRows(): IntArray = IntArray(rows) { i ->
        val row = data[i]
        row.indices.maxByOrNull { row[it].scalarValue } ?: 0
    }

    companion object {
        fun fromArray(array: Array<Array<Value>>): Matrix = Matrix(array)

        fun zeros(rows: Int, cols: Int): Matrix {
            val data = Array(rows) { Array(cols) { Value.ZERO } }
            return Matrix(data)
        }
    }
}
