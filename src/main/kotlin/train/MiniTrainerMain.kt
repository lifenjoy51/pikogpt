package train

/**
 * Scalar 백엔드 quickstart 학습 진입점.
 *
 * 학습자가 처음부터 끝까지 한 번 돌려볼 수 있는 가장 작은 설정. 알파벳 a-z 텍스트로
 * 음절·단어 패턴을 학습합니다.
 *
 * 선행 단계:
 *   ./gradlew runAlphabetPrep   # data/alphabet/{train.bin, val.bin, meta.json} 생성
 *
 * 실행:
 *   ./gradlew runMiniTrainer
 *
 * 산출물:
 *   model/alphabet/main/v0001/  (best 갱신 시 v0002, v0003, ...)
 *
 * 모델: layers=1, heads=2, embd=8, blockSize=16 (~수천 파라미터)
 * 학습: maxIters=1000, batchSize=16, lr=1e-3
 * 시간: Apple Silicon 기준 약 8~12분
 *
 * 학습 후 샘플링:
 *   ./gradlew runSampler                                # 자동 검색 (model/alphabet/main 최신 v)
 *   ./gradlew runSampler --args="model/alphabet/main/v0001"
 *
 * 자세한 한 흐름 가이드: docs/scalar-quickstart.md
 */
fun main() {
    val config = TrainConfig(
        // I/O
        dataPath = "data/alphabet",
        modelDir = "model",
        expName = "main",

        // 미니배치
        gradientAccumulationSteps = 1,
        batchSize = 16,
        blockSize = 16,

        // 모델 (가장 작은 사이즈 — 빠른 학습 + 알파벳 패턴 학습 가능)
        embeddingDimension = 8,
        numberOfLayers = 1,
        numberOfHeads = 2,
        bias = true,
        dropout = 0.1f,

        // 옵티마이저
        learningRate = 1.0e-3f,
        maxIters = 1000,
        weightDecay = 0.01f,
        gradClip = 1.0f,

        // 학습률 스케줄
        warmupRatio = 0.05f,
        learningRateDecayRatio = 0.95f,
        minimumLearningRate = 1e-5f,
        decayLr = true,

        // 평가/체크포인트
        evalIntervalRatio = 0.1f,   // 매 100 iter eval
        evalIters = 5,
        logInterval = 50,
        alwaysSaveCheckpoint = true,
    )
    ScalarTrainer(config).train()
}
