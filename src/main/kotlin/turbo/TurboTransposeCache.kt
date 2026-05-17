package turbo

import java.util.IdentityHashMap

/**
 * Weight transpose 결과 캐시. 한 학습 step 내 같은 weight tensor가 여러 micro-batch
 * (config.gradientAccumulationSteps × batchSize 회)에서 반복 사용되지만 step 내 값이
 * 변하지 않는다 — 같은 TurboTensor identity면 transpose 결과 재사용해 매번의 alloc +
 * memcpy를 한 번으로 줄인다.
 *
 * 만료 시점: [TurboTrainer]가 optimizer.step() 직후 [clear] 호출. AdamW가 weight를
 * in-place 갱신하므로 그 시점에 cache 무효화 안 하면 stale transpose를 반환한다.
 *
 * TurboParallel worker는 worker별 독립 TurboPikoGPT (즉 weight TurboTensor도 별개
 * 객체)라 IdentityHashMap이 worker별로 분리된 entry를 유지 — 충돌 없음.
 *
 * Thread-safety: TurboParallel ForkJoinPool worker가 동시에 transposeOf 호출 가능.
 * synchronized 블록으로 보호. cache hit이 dominant라 lock 경합은 짧다.
 */
object TurboTransposeCache {
    private val cache = IdentityHashMap<TurboTensor, TurboTensor>()

    fun transposeOf(t: TurboTensor): TurboTensor {
        synchronized(cache) {
            val hit = cache[t]
            if (hit != null) return hit
        }
        val transposed = t.transpose2D()
        synchronized(cache) {
            val existing = cache[t]
            if (existing != null) return existing
            cache[t] = transposed
            return transposed
        }
    }

    fun clear() {
        synchronized(cache) { cache.clear() }
    }
}
