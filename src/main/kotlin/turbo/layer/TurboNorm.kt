package turbo.layer

import turbo.TurboTensor

/**
 * 정규화 레이어 공통 인터페이스. Phase 1에서 LayerNorm/RMSNorm 분기에 사용.
 *
 * 한 모델 안의 모든 norm은 같은 타입이라 call site는 monomorphic — JIT가 inline하기 쉬움.
 */
sealed interface TurboNorm {
    fun forward(x: TurboTensor): TurboTensor
    fun backward(gy: TurboTensor): TurboTensor
    fun parameters(): List<TurboTensor>
}

/**
 * normalizationType 문자열로 적절한 norm 인스턴스 생성.
 *   "layernorm" → TurboLayerNorm
 *   "rmsnorm"   → TurboRMSNorm (β 없음)
 */
internal fun createTurboNorm(
    dim: Int,
    useBias: Boolean,
    normalizationType: String,
    eps: Float = 1e-5f,
): TurboNorm = when (normalizationType.lowercase()) {
    "rmsnorm" -> TurboRMSNorm(dim, eps)
    "layernorm" -> TurboLayerNorm(dim, useBias, eps)
    else -> error("Unknown normalizationType: $normalizationType (expected 'layernorm' or 'rmsnorm')")
}
