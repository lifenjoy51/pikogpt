package turbo

import gpt.GPTConfig
import kotlinx.serialization.Serializable

/**
 * turbo 백엔드의 모델 설정. `gpt.GPTConfig`(vec/scalar와 공유)를 wrap하고,
 * turbo 전용 알고리즘 옵션(RMSNorm, GQA, qk-norm, fused QKV, z-loss)을 추가한다.
 *
 * 모든 turbo 전용 옵션은 default OFF — 옵션을 켜지 않으면 Phase 0 (vec 동등성) 경로 그대로.
 */
@Serializable
data class TurboModelConfig(
    /** GPT 기본 아키텍처 — vec/scalar와 공유되는 부분. */
    val gpt: GPTConfig,

    /**
     * Normalization 종류 — `"layernorm"`(default, vec와 동등) 또는 `"rmsnorm"`(Llama 표준).
     * RMSNorm은 평균 중심화/β 없이 RMS만 정규화 → 파라미터 절반, ~7% 빠름.
     */
    val normalizationType: String = "layernorm",

    /**
     * GQA — KV head 수. `null`이면 MHA (numKvHeads = numberOfAttentionHeads).
     * `numKvHeads < numberOfAttentionHeads`이면 K/V가 head 그룹별로 broadcast됨.
     * `numberOfAttentionHeads`는 `numKvHeads`의 정수배여야 함.
     */
    val numKvHeads: Int? = null,

    /**
     * Q/K projection 직후 RMSNorm 적용. Attention score 폭주 억제.
     * 작은 모델은 효과 미미, 큰 모델/긴 컨텍스트에서 효과 큼.
     */
    val useQkNorm: Boolean = false,

    /**
     * QKV projection을 단일 matmul로 fused 처리 (Q, K, V 3개 Linear → 1개 Linear + slice).
     * 메모리 통과 횟수 감소 + JIT 친화. GQA 시 출력은 `embedDim + 2*kvDim`.
     */
    val useFusedQkv: Boolean = false,

    /**
     * Cross-entropy z-loss 가중치. `loss += zLossWeight * mean(lse²)`.
     * 0이면 비활성. 0.0001 정도가 표준 (PaLM/T5).
     */
    val zLossWeight: Float = 0.0f,
) {
    /** 모든 옵션이 OFF인 경우만 Phase 0 회귀 경로 (vec 동등성 검증 가능). */
    val isPhase0Compatible: Boolean
        get() = normalizationType.equals("layernorm", ignoreCase = true) &&
                numKvHeads == null &&
                !useQkNorm &&
                !useFusedQkv &&
                zLossWeight == 0.0f

    /** 실효 KV head 수 — null이면 MHA로 numHeads 그대로. */
    val effectiveKvHeads: Int
        get() = numKvHeads ?: gpt.numberOfAttentionHeads
}
