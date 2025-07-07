package data

class ShakespeareBPEPrepTest {

    fun main() {
        // BPE 토크나이저로 데이터 준비
        val prep = ShakespeareBPEPrep()
        prep.prepare()

        // 저장된 데이터 로드 테스트
        println("\n=== 데이터 로드 테스트 ===")
        val loader = BPEDataLoader()
        val trainIds = loader.loadBinary("train.bin")
        println("로드된 훈련 토큰 수: ${"%,d".format(trainIds.size)}")
        println("처음 20개 토큰: ${trainIds.take(20).joinToString(" ")}")
    }
}