package mps

/**
 * MPSGraph 모델 graph build에 필요한 hyperparam. TurboTrainConfig의 학습 hyperparam과는 분리.
 *
 * graph는 batch/blockSize 등 모든 차원이 빌드 시점에 고정. iter마다 동일 graph 재실행.
 */
data class MpsGraphConfig(
    val numLayers: Int,
    val embedDim: Int,
    val numHeads: Int,
    val blockSize: Int,
    val vocab: Int,
    val batchSize: Int,
    val useSwiglu: Boolean = true,
    val useRope: Boolean = true,
    val tieWeights: Boolean = true,
    /**
     * P2.1 — stepGraph forward를 fp16으로 계산. weight/grad/m/v는 fp32 유지 (master copy).
     *
     * 분기 위치: stepGraph build 시점 (cast on entry, cast back on exit).
     *   - 모든 weight placeholder를 fp16 cast → forward computation fp16
     *   - logits을 fp32 cast → CE loss는 fp32 (안정성)
     *   - backward는 fp32 weight target (chain rule이 cast op 통해 자동 backprop)
     *
     * 안정성 위험:
     *   - LayerNorm fp16에서 mean/variance 정확도 손해 가능
     *   - softmax fp16에서 logit overflow 가능
     *   - loss scaling 없음 (PoC; grad underflow는 cast back에서 자연 보장)
     *
     * 단위 test (MpsGraphFp16Test)에서 fp32 baseline 대비 loss diff < 0.05 검증.
     */
    val useFp16: Boolean = false,
    /**
     * P1.3 — stepGraph에서 MPSGraph Variable API를 사용해 weight/m/v를 graph 내부에 보관하고
     * assignVariable로 in-place update한다.
     *
     * true 시점:
     *   - 매 step weight/m/v feeds 생략 (variable이 자체 storage)
     *   - result에 weight 안 받음 (in-place)
     *   - ping-pong swap 없음
     *   - readWeight/loadWeights는 별도 read/write graph로 처리 (cache됨)
     *
     * false (default) 시점:
     *   - 기존 placeholder + slot.buffer/alt ping-pong path
     *   - accum/adam graph와 cross-graph sync 필요 없음
     *
     * true 모드에서는 runAccumStep/runAdamStep 호출이 placeholder mode를 가정하므로 동작이 다를 수 있음.
     * stepGraph 단일 사용 가정.
     */
    val useVariableForStep: Boolean = false,
    /**
     * P4 — turbo 동등 dropout. true 시점:
     *   - graph build에 attention output + MLP output 후 dropout op 삽입
     *   - dropout은 host-side mask placeholder 방식 (forward/backward 같은 mask로 정확한 chain rule)
     *   - eval mode (`runForwardLoss`)에서는 mask=1 (identity)로 host에서 강제
     *   - 학습 mode (`runAccumStep`/`runTrainingStep`)에서는 매 step 새 mask 생성
     * dropoutProbability는 0..1 — 0이면 mask=1 강제 (효과 없음, useDropout=true 의미만 표시).
     */
    val useDropout: Boolean = false,
    val dropoutProbability: Float = 0.0f,
)
