package train.experiments

import mps.MpsBackend
import turbo.TurboTrainConfig
import turbo.TurboTrainer

/**
 * F PoC — 10M 모델 forward fp16 mixed precision. backward는 fp32 그대로 (안전 마진).
 *
 * Bench10MMpsTurbo와 동일 config + `MpsBackend.enableFp16()` 한 줄 추가, `expName="bench10m-mps-fp16"`.
 *
 * 검증 권장: 첫 100 iter loss curve가 Bench10MMpsTurbo와 ±0.02 이내인지 확인 후 채택.
 * fp16 누적 오차(특히 RMSNorm/SwiGLU)로 인한 loss spike/NaN 가능성 — 발생 시 폐기.
 */
fun main(args: Array<String>) {
    val mpsOk = MpsBackend.enable()
    if (!mpsOk) {
        println("[mps] disabled — fp16 PoC도 의미 없음. turbo CPU fallback.")
    } else {
        MpsBackend.enableFp16()
    }

    val maxIters = args.firstOrNull()?.toIntOrNull() ?: 250

    val config = TurboTrainConfig(
        dataPath = "data/ccmc-v2-pro/stage2",
        modelDir = "model",
        expName = "bench10m-mps-fp16",
        gradientAccumulationSteps = 16,
        batchSize = 2,
        blockSize = 32,
        numberOfLayers = 16,
        numberOfHeads = 8,
        embeddingDimension = 256,
        dropout = 0.0f,
        bias = true,
        tieWeights = true,
        mlpActivation = "swiglu",
        positionEncoding = "rope",
        learningRate = 3e-4f,
        weightDecay = 0.01f,
        labelSmoothing = 0.0f,
        gradClip = 1.0f,
        beta1 = 0.9f,
        beta2 = 0.95f,
        maxIters = maxIters,
        warmupRatio = 0.05f,
        learningRateDecayRatio = 0.95f,
        minimumLearningRate = 3e-5f,
        decayLr = true,
        evalIntervalRatio = 0.4f,
        evalIters = 25,
        logInterval = 50,
        alwaysSaveCheckpoint = false,
        initFrom = "scratch",
    )

    TurboTrainer(config).train()
}
