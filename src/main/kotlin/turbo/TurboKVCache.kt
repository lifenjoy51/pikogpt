package turbo

/**
 * Per-layer K/V ring buffer — sampler incremental decode 가속 (Phase 3.0).
 *
 * 원리:
 *   토큰 t를 생성할 때 이전 0..t-1의 Q/K/V를 다시 계산하지 않고, 새 토큰의 K_t, V_t만
 *   계산해 cache에 추가. attention은 Q_t · K[0..t]^T → softmax → · V[0..t].
 *
 *   토큰당 비용:
 *     - naive forward: O(t × C × C) (전체 시퀀스 forward)
 *     - KV cache:     O(C × C) Q/K/V projection + O(t × C) attention
 *
 * 사용 패턴 (sampler):
 *   ```
 *   val cache = TurboKVCache(maxSeqLen, numLayers, kvDim)
 *   for (token in promptTokens + generatedTokens) {
 *     val logits = model.forwardIncremental(token, cache)
 *     val next = sample(logits)
 *   }
 *   ```
 *
 * 모든 layer가 같은 timestep을 순서대로 처리한다고 가정. layer N-1 처리 후 자동으로
 * len이 1 증가 (다음 토큰 받을 준비).
 */
class TurboKVCache(
    val maxSeqLen: Int,
    val numLayers: Int,
    val kvDim: Int,
) {
    private val kBuf: Array<FloatArray> = Array(numLayers) { FloatArray(maxSeqLen * kvDim) }
    private val vBuf: Array<FloatArray> = Array(numLayers) { FloatArray(maxSeqLen * kvDim) }

    /** 누적된 토큰 수 (이미 모든 layer로 처리 완료된 timestep 개수). */
    private var len: Int = 0

    /** 다음에 쓰일 layer 인덱스 — 0이면 새 timestep 시작, numLayers-1 처리 후 0으로 회귀하며 len++. */
    private var nextLayer: Int = 0

    /** 현재 처리 중인 토큰의 position (RoPE에 사용). */
    val currentPosition: Int get() = len

    /** append 호출 후 attention이 사용할 sequence length (len+1). */
    val lengthAfterAppend: Int get() = len + 1

    /** 누적된 토큰 수. */
    val length: Int get() = len

    /**
     * layer의 K_t, V_t를 buffer에 추가. layer 0..numLayers-1 순서로 호출되어야 한다.
     * 모든 layer가 처리한 후에야 len 증가.
     *
     * @return 이 layer가 attention 계산에 사용할 수 있는 sequence length (= 이전 토큰들 + 현재 토큰).
     */
    fun append(layer: Int, kRow: FloatArray, vRow: FloatArray): Int {
        require(layer == nextLayer) { "out-of-order layer write: expected $nextLayer, got $layer" }
        require(kRow.size == kvDim && vRow.size == kvDim) { "kRow/vRow size != kvDim" }
        require(len < maxSeqLen) { "KV cache full: len=$len, max=$maxSeqLen" }
        val pos = len  // 현재 토큰의 position (0-indexed)
        val dst = pos * kvDim
        kRow.copyInto(kBuf[layer], destinationOffset = dst)
        vRow.copyInto(vBuf[layer], destinationOffset = dst)
        nextLayer++
        if (nextLayer == numLayers) {
            nextLayer = 0
            len++
        }
        return pos + 1  // attention 계산용 sequence length
    }

    fun getKBuffer(layer: Int): FloatArray = kBuf[layer]
    fun getVBuffer(layer: Int): FloatArray = vBuf[layer]

    fun reset() {
        len = 0
        nextLayer = 0
    }
}
