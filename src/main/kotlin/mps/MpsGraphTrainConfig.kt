package mps

import kotlinx.serialization.Serializable

/**
 * P0.5 — MPSGraph backend 학습 설정.
 *
 * TurboTrainConfig schema 일부를 그대로 따른다 (체크포인트 호환 + 진입점 친숙도).
 * batchSize는 P1.1 (batch 일반화) 전까지 1로 강제.
 */
@Serializable
data class MpsGraphTrainConfig(
    // I/O
    val dataPath: String,
    val modelDir: String = "model",
    val expName: String = "main",
    val alwaysSaveCheckpoint: Boolean = true,
    /** 'scratch': 처음부터, 'resume': 최신 ckpt에서 이어. */
    val initFrom: String = "scratch",

    // 데이터
    val blockSize: Int = 32,
    /** P1.1 — micro-batch size. 1 → 기존 PoC scope. >1이면 step graph가 [B, T] 처리. */
    val batchSize: Int = 1,
    /**
     * P1.2 — gradient accumulation steps. effective batch = batchSize × gradientAccumulationSteps.
     * 현재 PoC: 매 micro-step에 AdamW 적용 (진정한 grad accum은 P1.3 Variable 패러다임과 함께).
     * 1보다 크면 trainer가 매 iter마다 N micro-step을 실행 + 마지막에 1 effective iter로 카운트.
     */
    val gradientAccumulationSteps: Int = 1,

    // 옵티마이저
    val learningRate: Float = 3e-4f,
    val maxIters: Int = 5000,
    val weightDecay: Float = 0.01f,
    val beta1: Float = 0.9f,
    val beta2: Float = 0.95f,
    val eps: Float = 1e-8f,
    /** Global gradient norm clip. 0이면 disabled. */
    val gradClip: Float = 1.0f,

    // 스케줄
    val decayLr: Boolean = true,
    val warmupRatio: Float = 0.01f,
    val learningRateDecayRatio: Float = 0.8f,
    val minimumLearningRate: Float = 1e-5f,

    // 평가
    val evalIntervalRatio: Float = 0.01f,
    val logInterval: Int = 10,
    val evalIters: Int = 10,

    // Early stop
    val earlyStopPatience: Int = 0,

    /**
     * P2.1 — fp16 mixed precision. true 시점:
     *   - stepGraph forward를 fp16으로 계산 (matmul / SwiGLU / attention 모두 fp16)
     *   - LN은 안정성 위해 fp32 cast (entry/exit)
     *   - logits/CE loss/AdamW/grad/master weight는 fp32
     *   - backward target은 fp32 weight (autograd가 cast op 통해 chain rule 적용)
     * 단위 test (MpsGraphFp16Test)에서 fp32 baseline 대비 diff < 0.05 검증됨.
     */
    val useFp16: Boolean = false,

    /**
     * P4 — turbo 동등 dropout. 0..1. 0이면 dropout 없음.
     * 활성화 시 graph build 시점에 attention output + MLP output 후 dropout op 삽입.
     * 학습 trainer가 매 micro-step random mask 생성 (inverted dropout, keep=1/(1-p), drop=0).
     * eval (`runForwardLoss`)에서는 dropout 적용 안 함 (forward-only graph는 placeholder 없음).
     */
    val dropoutProbability: Float = 0.0f,

    /**
     * P3.1 — MPSGraphExecutable compile + 디스크 serialize 활용.
     * true + executableCachePath 지정 시 trainer가 처음 build 후 compile + serialize 시도하고,
     * 다음 session에서 같은 path가 있으면 deserialize. 본 PoC는 roundtrip 검증 완료;
     * run path 자체는 graph 기반 유지 (executable.runWithMTLCommandQueue inputsArray ordering
     * refactor는 별도 작업).
     */
    val useExecutableCache: Boolean = false,
    val executableCachePath: String? = null,
) {
    val warmupIters: Int get() = (maxIters * warmupRatio).toInt().coerceAtLeast(1)
    val learningRateDecayIterations: Int get() = (maxIters * learningRateDecayRatio).toInt().coerceAtLeast(warmupIters + 1)
    val evalInterval: Int get() = (maxIters * evalIntervalRatio).toInt().coerceAtLeast(1)
}
