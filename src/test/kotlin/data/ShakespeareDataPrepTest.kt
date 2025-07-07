package data

class ShakespeareDataPrepTest {

    // 사용 예시
    fun main() {
        // 데이터 준비
        val prep = ShakespeareCharPrep()
        prep.prepare()

        // 데이터 로드 테스트
        println("\n=== 데이터 로드 테스트 ===")
        val loader = DataLoader()

        // 메타 정보 로드
        val meta = loader.loadMetaInfo("meta.json")
        println("어휘 크기: ${meta.vocabSize}")

        // 훈련 데이터 일부 로드
        val trainIds = loader.loadBinary("train.bin")
        println("훈련 데이터 처음 100개 토큰:")
        println(trainIds.take(100).joinToString(" "))

        // 디코딩 테스트
        val sampleIds = trainIds.take(100).toList()
        val decoded = sampleIds.joinToString("") { meta.itos[it]!! }
        println("\n디코딩된 텍스트:")
        println(decoded)
    }

}