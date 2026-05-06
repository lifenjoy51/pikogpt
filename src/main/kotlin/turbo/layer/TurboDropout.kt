package turbo.layer

import turbo.TurboTensor
import kotlin.random.Random

/**
 * Inverted dropout. Phase 0은 vec와 동일.
 *   training=true & p>0:  mask = 1/(1-p) with prob (1-p), 0 with prob p, y = x ⊙ mask
 *   else: identity
 */
class TurboDropout(val probability: Float) {
    var training: Boolean = true

    private var cachedMask: FloatArray? = null
    private var cachedShape: IntArray? = null

    fun forward(x: TurboTensor): TurboTensor {
        if (!training || probability <= 0.0f) {
            cachedMask = null
            return x
        }
        val keepScale = 1.0f / (1.0f - probability)
        val mask = FloatArray(x.numel)
        for (i in mask.indices) {
            mask[i] = if (Random.Default.nextFloat() < probability) 0.0f else keepScale
        }
        cachedMask = mask
        cachedShape = x.shape

        val out = TurboTensor(x.shape.copyOf())
        for (i in out.data.indices) out.data[i] = x.data[i] * mask[i]
        return out
    }

    fun backward(gy: TurboTensor): TurboTensor {
        val mask = cachedMask ?: return gy
        require(gy.numel == mask.size) { "backward gy shape != forward mask shape" }
        val out = TurboTensor(gy.shape.copyOf())
        for (i in out.data.indices) out.data[i] = gy.data[i] * mask[i]
        return out
    }

    fun parameters(): List<TurboTensor> = emptyList()
}
