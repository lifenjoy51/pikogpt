package mps

import kotlin.math.PI
import kotlin.math.cos

/**
 * P0.2 — Warmup + cosine decay LR scheduler.
 *
 * `TurboTrainer.getLearningRate`와 동일한 공식. step graph는 lr placeholder로 노출되어 있어
 * graph rebuild 없이 매 iter host에서 새 lr 값을 주입할 수 있다.
 *
 *   iter < warmupIters     → 선형 warmup ((iter+1)/warmup) * baseLr
 *   iter > decayIters      → minLr
 *   사이                    → minLr + 0.5*(1+cos(π*r)) * (baseLr - minLr),  r = (iter-warmup)/(decay-warmup)
 */
object MpsLrSchedule {
    fun computeLr(
        iter: Int,
        warmupIters: Int,
        decayIters: Int,
        baseLr: Float,
        minLr: Float,
        decayEnabled: Boolean = true,
    ): Float {
        if (!decayEnabled) return baseLr
        if (iter < warmupIters) {
            return baseLr * (iter + 1).toFloat() / warmupIters.coerceAtLeast(1).toFloat()
        }
        if (iter > decayIters) return minLr
        val span = (decayIters - warmupIters).coerceAtLeast(1)
        val decayRatio = (iter - warmupIters).toDouble() / span
        val coefficient = 0.5f * (1.0f + cos(PI * decayRatio).toFloat())
        return minLr + coefficient * (baseLr - minLr)
    }
}
