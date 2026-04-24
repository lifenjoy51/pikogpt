package vec.ops

import vec.Tensor
import vec.productInt

/**
 * 2차원 행렬 곱. 수식과 backward를 한곳에 모아 둔 연산.
 *
 *   Shape:     A[M, K] · B[K, N] → C[M, N]
 *   Forward:   C[i, j] = Σ_k A[i, k] * B[k, j]
 *   Backward:  ∂L/∂A = ∂L/∂C · B^T     (shape [M, K])
 *              ∂L/∂B = A^T · ∂L/∂C     (shape [K, N])
 *
 * backward는 A/B의 grad 배열에 **더한다** (accumulate). 학습 루프가 iter 시작에
 * `zeroGrad`를 부르기 전까지 누적된 값이 유지되는 관례에 맞춘다.
 */
fun matmul(a: Tensor, b: Tensor): Tensor {
    require(a.shape.size == 2 && b.shape.size == 2) {
        "matmul은 2차원 텐서만: a=${a.shape.contentToString()}, b=${b.shape.contentToString()}"
    }
    val m = a.rows
    val k = a.cols
    val n = b.cols
    require(b.rows == k) { "차원 불일치: A[$m, $k] · B[${b.rows}, $n]" }

    // "ikj" 루프 순서 — inner loop에서 B와 C의 row를 연속으로 접근해 캐시 친화적.
    // 수학적으로 "ijk"와 동일 (accumulation 순서만 다름, float 누적 차이는 미세).
    val c = Tensor(intArrayOf(m, n))  // FloatArray 기본값 0 — += 누적용으로 그대로 사용
    for (i in 0 until m) {
        val aRowOffset = i * k
        val cRowOffset = i * n
        for (kk in 0 until k) {
            val aik = a.data[aRowOffset + kk]
            val bRowOffset = kk * n
            for (j in 0 until n) {
                c.data[cRowOffset + j] += aik * b.data[bRowOffset + j]
            }
        }
    }
    return c
}

/**
 * matmul의 backward. C = A · B일 때 출력에 대한 기울기 `gy` (shape [M, N])를 받아
 *   ∂L/∂A += gy · B^T      (A의 grad에 누적)
 *   ∂L/∂B += A^T · gy      (B의 grad에 누적)
 * 를 수행한다.
 *
 * `gyData`를 별도로 받는 이유: Tensor의 grad 필드 외에 "외부에서 만든 기울기"를
 * 그대로 넘기는 경우도 있어 (예: loss의 초기 기울기) 유연하게 한다.
 */
fun matmulBackward(a: Tensor, b: Tensor, gyData: FloatArray) {
    val m = a.rows
    val k = a.cols
    val n = b.cols
    require(gyData.size == m * n) { "gy 크기 불일치: expected ${m * n}, got ${gyData.size}" }

    val dA = a.gradOrAlloc()
    val dB = b.gradOrAlloc()

    // ∂L/∂A[i, kk] += Σ_j gy[i, j] * B[kk, j]
    for (i in 0 until m) {
        for (kk in 0 until k) {
            var sum = 0.0f
            for (j in 0 until n) {
                sum += gyData[i * n + j] * b.data[kk * n + j]
            }
            dA[i * k + kk] += sum
        }
    }

    // ∂L/∂B[kk, j] += Σ_i A[i, kk] * gy[i, j]
    for (kk in 0 until k) {
        for (j in 0 until n) {
            var sum = 0.0f
            for (i in 0 until m) {
                sum += a.data[i * k + kk] * gyData[i * n + j]
            }
            dB[kk * n + j] += sum
        }
    }
}
