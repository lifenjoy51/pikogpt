package mps

import mps.ops.mpsMatmul
import mps.ops.mpsMatmulBackward
import mps.ops.mpsMatmulFp16
import turbo.ops.matmulBackwardImpl
import turbo.ops.matmulImpl

/**
 * MatMul dispatch를 Metal로 교체/복원. 학습 진입점에서 `enable()` 한 번 호출.
 *
 *   - macOS arm64 + dylib + Metal init 성공 → matmulImpl/matmulBackwardImpl을 mps* 로 교체
 *   - 그 외(미macOS, dylib 없음, init 실패) → turbo 유지 + 사유 출력
 *
 * `disable()`은 테스트/벤치마크에서 turbo로 되돌릴 때 사용.
 */
object MpsBackend {

    @Volatile var enabled: Boolean = false
        private set

    fun enable(): Boolean {
        if (enabled) return true
        if (!MpsAvailability.ensureChecked()) {
            println("[mps] backend NOT enabled: ${MpsAvailability.reason}")
            return false
        }
        matmulImpl = { a, b -> mpsMatmul(a, b) }
        matmulBackwardImpl = { a, b, gy -> mpsMatmulBackward(a, b, gy) }
        // GPU는 single command queue 직렬 실행 → TurboParallel 데이터 병렬(worker 다수)과 충돌.
        // 8 worker가 같은 queue에 dispatch하면 GPU는 직렬 처리하면서 CPU SIMD × 8 worker에게 패배.
        // 강제로 worker=1로 두고 single 경로에서 GPU dispatch 비용을 큰 matmul로 상쇄.
        System.setProperty("TURBO_MAX_WORKERS", "1")
        enabled = true
        println("[mps] backend enabled — ${MpsAvailability.reason}")
        println("[mps] TurboParallel 비활성 강제 — TURBO_MAX_WORKERS=1 (GPU queue 직렬화 회피)")
        return true
    }

    fun disable() {
        matmulImpl = { a, b -> turbo.ops.turboMatmul(a, b) }
        matmulBackwardImpl = { a, b, gy -> turbo.ops.turboMatmulBackward(a, b, gy) }
        enabled = false
    }

    /**
     * F: fp16 mixed precision 활성. enable() 후에만 의미 있음.
     *
     * forward만 fp16 ([mpsMatmulFp16]), backward는 fp32 그대로. 학습 안정성은
     * 사용자가 첫 100 iter loss curve로 검증 후 채택해야 한다 (NaN/loss spike 위험).
     *
     * 예상 효과: 큰 matmul에서 추가 1.5~2× 가속. 정확도 fp32 대비 ~1e-3 오차.
     */
    fun enableFp16(): Boolean {
        if (!enabled) {
            println("[mps] enableFp16 무시: MpsBackend.enable() 먼저 호출 필요")
            return false
        }
        matmulImpl = { a, b -> mpsMatmulFp16(a, b) }
        println("[mps] fp16 forward 활성 — 학습 안정성 첫 100 iter 모니터링 권장")
        return true
    }
}
