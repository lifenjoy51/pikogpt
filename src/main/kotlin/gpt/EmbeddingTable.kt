package gpt

import RandomGaussian
import Value

/**
 * 임베딩 테이블 클래스
 * 
 * 토큰 ID를 고정 크기 벡터로 매핑하는 룩업 테이블을 표현합니다.
 * 토큰 임베딩과 위치 임베딩 모두에 사용될 수 있습니다.
 * 
 * @param tableSize 테이블의 크기 (어휘 크기 또는 최대 시퀀스 길이)
 * @param embeddingDim 임베딩 벡터의 차원
 * @param initStd 가중치 초기화 시 표준편차
 */
class EmbeddingTable : Matrix {
    val tableSize: Int
    val embeddingDim: Int
    
    constructor(
        tableSize: Int,
        embeddingDim: Int,
        initStd: Float = 0.02f
    ) : super(Array(tableSize) {
        Array(embeddingDim) { Value((RandomGaussian.next() * initStd).toFloat()) }
    }) {
        this.tableSize = tableSize
        this.embeddingDim = embeddingDim
    }
    
    private constructor(data: Array<Array<Value>>) : super(data) {
        this.tableSize = data.size
        this.embeddingDim = if (data.isNotEmpty()) data[0].size else 0
    }
    
    /**
     * 특정 인덱스의 임베딩 벡터 가져오기
     * @param index 인덱스 (0부터 시작)
     * @return 해당 인덱스의 임베딩 벡터
     */
    override operator fun get(index: Int): Array<Value> {
        require(index in 0 until tableSize) { 
            "인덱스가 범위를 벗어났습니다: $index (유효 범위: 0 until $tableSize)" 
        }
        return data[index]
    }
    
    /**
     * 특정 인덱스의 임베딩 벡터 설정
     * @param index 인덱스 (0부터 시작)
     * @param embedding 설정할 임베딩 벡터
     */
    operator fun set(index: Int, embedding: Array<Value>) {
        require(index in 0 until tableSize) { 
            "인덱스가 범위를 벗어났습니다: $index (유효 범위: 0 until $tableSize)" 
        }
        require(embedding.size == embeddingDim) {
            "임베딩 차원이 맞지 않습니다: ${embedding.size} (예상: $embeddingDim)"
        }
        data[index] = embedding
    }
    
    /**
     * 여러 인덱스에 대한 임베딩 벡터들을 한번에 조회
     * @param indices 인덱스 배열
     * @return 해당 인덱스들의 임베딩 시퀀스
     */
    fun lookup(indices: IntArray): Sequence {
        val embeddings = Array(indices.size) { i ->
            get(indices[i])
        }
        return Sequence.fromArray(embeddings)
    }
    
    /**
     * 단일 인덱스에 대한 임베딩 벡터 조회 (시퀀스로 반환)
     * @param index 인덱스
     * @return 1x1 시퀀스 (단일 임베딩)
     */
    fun lookupSingle(index: Int): Sequence {
        return lookup(intArrayOf(index))
    }
    
    
    
    companion object {
        /**
         * 기존 배열에서 임베딩 테이블 생성
         * @param array 기존 2차원 배열
         * @return 임베딩 테이블 객체
         */
        fun fromArray(array: Array<Array<Value>>): EmbeddingTable {
            return EmbeddingTable(array)
        }
    }
}