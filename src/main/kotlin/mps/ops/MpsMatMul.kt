package mps.ops

import mps.jni.MetalMatMulBridge
import turbo.TurboTensor
import turbo.ops.turboMatmul
import turbo.ops.turboMatmulBackward

/**
 * mps가 turbo보다 빨라지는 작업량 임계값. M*K*N (총 곱셈 횟수, FLOPs/2) 기준.
 *
 * 2026-05-17 microbench 측정 (10M 모델 실제 shape):
 *   - [64, 256, 256]  = 4.2M ops → mps 0.98× (살짝 느림 — turbo로 fallback)
 *   - [128, 256, 256] = 8.4M ops → mps 2.63×
 *   - [64, 256, 1024] = 16.8M ops → mps 3.84×
 *
 * 8M ops 미만은 GPU dispatch overhead가 compute 시간보다 커져 turbo SIMD가 우위.
 * 측정 시점 변경되면 이 상수도 재조정.
 */
private const val MPS_MIN_OPS: Long = 8_000_000L

private fun shouldUseTurbo(m: Int, k: Int, n: Int): Boolean {
    val ops = m.toLong() * k.toLong() * n.toLong()
    return ops < MPS_MIN_OPS
}

/**
 * Metal GPU MatMul (forward + backward). 시그니처는 `turbo.ops.turboMatmul`과 동일.
 *
 *   Shape:     A[M, K] · B[K, N] → C[M, N]
 *   Forward:   C[i, j]  = Σ_k A[i, k] * B[k, j]
 *   Backward:  dA[i, k] += Σ_j gy[i, j] * B[k, j]
 *              dB[k, j] += Σ_i A[i, k] * gy[i, j]
 *
 * turbo와 동일하게 `dA`/`dB`는 **누적**(+=). float32 누적 순서가 turbo와 다를 수 있으나
 * 같은 inner sum을 lane-wise/thread-wise 재배치한 것이라 결과는 epsilon 이내.
 */
fun mpsMatmul(a: TurboTensor, b: TurboTensor): TurboTensor {
    require(a.shape.size == 2 && b.shape.size == 2) {
        "mpsMatmul은 2차원 텐서만: a=${a.shape.contentToString()}, b=${b.shape.contentToString()}"
    }
    val m = a.rows
    val k = a.cols
    val n = b.cols
    require(b.rows == k) { "차원 불일치: A[$m, $k] · B[${b.rows}, $n]" }

    // 작은 shape은 GPU dispatch overhead로 turbo SIMD가 우위 → 자동 fallback.
    if (shouldUseTurbo(m, k, n)) return turboMatmul(a, b)

    val c = TurboTensor(intArrayOf(m, n))
    MetalMatMulBridge.nativeMatmul(a.data, b.data, m, k, n, c.data)
    return c
}

fun mpsMatmulBackward(a: TurboTensor, b: TurboTensor, gyData: FloatArray) {
    val m = a.rows
    val k = a.cols
    val n = b.cols
    require(gyData.size == m * n) { "gy 크기 불일치: expected ${m * n}, got ${gyData.size}" }

    if (shouldUseTurbo(m, k, n)) {
        turboMatmulBackward(a, b, gyData)
        return
    }

    val dA = a.gradOrAlloc()
    val dB = b.gradOrAlloc()

    MetalMatMulBridge.nativeMatmulBackwardA(b.data, gyData, m, k, n, dA)
    MetalMatMulBridge.nativeMatmulBackwardB(a.data, gyData, m, k, n, dB)
}

/**
 * fp16 mixed precision forward. backward는 [mpsMatmulBackward](fp32) 그대로 사용 권장.
 *
 * 학습 안정성 위험 (특히 RMSNorm/SwiGLU에서 fp16 누적 오차 누적) — 기본 비활성.
 * [mps.MpsBackend.enableFp16] 호출 시 matmulImpl이 이 함수로 교체된다.
 *
 * 정확도: fp16 mma는 fp32 대비 ~1e-3 오차. [mps.MpsMatMulFp16Test]는 rtol 5e-3 통과.
 */
fun mpsMatmulFp16(a: TurboTensor, b: TurboTensor): TurboTensor {
    require(a.shape.size == 2 && b.shape.size == 2) {
        "mpsMatmulFp16은 2차원 텐서만: a=${a.shape.contentToString()}, b=${b.shape.contentToString()}"
    }
    val m = a.rows
    val k = a.cols
    val n = b.cols
    require(b.rows == k) { "차원 불일치: A[$m, $k] · B[${b.rows}, $n]" }

    // 작은 shape은 fp16 변환 비용 + dispatch overhead로 turbo SIMD 우위.
    if (shouldUseTurbo(m, k, n)) return turboMatmul(a, b)

    val c = TurboTensor(intArrayOf(m, n))
    MetalMatMulBridge.nativeMatmulFp16(a.data, b.data, m, k, n, c.data)
    return c
}
