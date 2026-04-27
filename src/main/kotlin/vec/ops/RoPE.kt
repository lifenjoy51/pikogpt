package vec.ops

import vec.Tensor
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin

/**
 * Rotary Position Embedding (RoPE) — Su et al. 2021.
 *
 * 학습 가능 position embedding 대신 Q와 K에 위치 의존 회전(rotation)을 적용해
 * relative position 정보를 attention dot product에 자연스럽게 주입.
 *
 * **수식** (head 내부, head_dim D, position pos, dim 인덱스 i ∈ [0, D/2)):
 *   theta_i = 10000^(-2i / D)
 *   angle   = pos * theta_i
 *   x'[i]        = x[i]    * cos(angle) - x[i+D/2] * sin(angle)
 *   x'[i + D/2]  = x[i]    * sin(angle) + x[i+D/2] * cos(angle)
 *
 * (페어링 방식은 GPT-NeoX 스타일 — `(i, i+D/2)` half-split. Llama의 `(2i, 2i+1)` interleave는
 * 수학적으로 동등하나 메모리 접근 패턴만 다름. 여기선 half-split이 구현·미분 모두 단순.)
 *
 * **backward**: 회전의 역은 `-angle`로 회전. 즉 transpose. 따라서:
 *   dx[i]        = dx'[i]    * cos + dx'[i+D/2] * sin
 *   dx[i + D/2]  = -dx'[i]   * sin + dx'[i+D/2] * cos
 *
 * 형태 가정: 입력 Tensor 는 [T, H*D] (T = seq len, H = numHeads, D = head_dim).
 * head_dim D는 **짝수**여야 (D/2 페어링 때문에).
 *
 * `applyRoPE`는 in-place로 회전을 적용해 Tensor를 직접 수정. 캐싱이 필요한 cos/sin 테이블은
 * 호출 시 매번 계산 — head_dim, T 작아 비용 무시 가능 (각 ~수천 곱셈).
 */
fun applyRoPE(x: Tensor, numHeads: Int) {
    require(x.shape.size == 2) { "RoPE 입력은 [T, H*D] 형태" }
    val t = x.rows
    val totalDim = x.cols
    val headDim = totalDim / numHeads
    require(totalDim % numHeads == 0) { "embedDim must be divisible by numHeads" }
    require(headDim % 2 == 0) { "head_dim must be even for RoPE (got $headDim)" }
    val half = headDim / 2

    for (pos in 0 until t) {
        for (h in 0 until numHeads) {
            val headOffset = h * headDim
            for (i in 0 until half) {
                val theta = 10000.0.pow(-2.0 * i / headDim)
                val angle = pos * theta
                val c = cos(angle).toFloat()
                val s = sin(angle).toFloat()

                val idxA = pos * totalDim + headOffset + i
                val idxB = pos * totalDim + headOffset + i + half
                val xa = x.data[idxA]
                val xb = x.data[idxB]
                x.data[idxA] = xa * c - xb * s
                x.data[idxB] = xa * s + xb * c
            }
        }
    }
}

/**
 * RoPE backward — `dx`를 in-place로 역회전(-angle)으로 변환.
 *
 *   dx_orig[i]        = dx_rotated[i]    * cos + dx_rotated[i+half] * sin
 *   dx_orig[i + half] = -dx_rotated[i]   * sin + dx_rotated[i+half] * cos
 *
 * 즉 forward와 같은 cos/sin이지만 sin 부호가 반전된 회전 적용.
 */
fun applyRoPEBackward(dx: Tensor, numHeads: Int) {
    require(dx.shape.size == 2)
    val t = dx.rows
    val totalDim = dx.cols
    val headDim = totalDim / numHeads
    require(headDim % 2 == 0)
    val half = headDim / 2

    for (pos in 0 until t) {
        for (h in 0 until numHeads) {
            val headOffset = h * headDim
            for (i in 0 until half) {
                val theta = 10000.0.pow(-2.0 * i / headDim)
                val angle = pos * theta
                val c = cos(angle).toFloat()
                val s = sin(angle).toFloat()

                val idxA = pos * totalDim + headOffset + i
                val idxB = pos * totalDim + headOffset + i + half
                val da = dx.data[idxA]
                val db = dx.data[idxB]
                // forward 회전 행렬 R(θ)의 transpose = R(-θ).
                dx.data[idxA] = da * c + db * s
                dx.data[idxB] = -da * s + db * c
            }
        }
    }
}
