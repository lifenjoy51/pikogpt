package turbo

import java.util.concurrent.ForkJoinPool

/**
 * 데이터 병렬 학습 유틸 (Phase 0은 vec.Parallel과 동일).
 * Phase 5.0: ForkJoinPool helper 추가. work-stealing 스케줄링으로 coroutine 대비 CPU bound
 * 핵심 루프에서 더 가벼운 dispatch. 일반 사용 패턴:
 *   ```
 *   turboForkJoinAll(workers.indices.toList()) { wi -> /* worker[wi] forward+backward */ }
 *   ```
 *
 * Phase 5.1+ 에서 worker-replica 방식 폐기 + thread-local activation pool + grad accumulator로
 * 재설계 (메모리 N배 → 1배 + 더 큰 scaling).
 */

/**
 * tasks를 ForkJoinPool.commonPool()에서 병렬 실행하고 모두 완료될 때까지 block.
 * commonPool은 JVM 시작 시 `Runtime.availableProcessors()` 크기로 lazy 초기화 — 자동.
 */
internal fun turboForkJoinAll(tasks: List<() -> Unit>) {
    val pool = ForkJoinPool.commonPool()
    val futures = tasks.map { task -> pool.submit { task() } }
    futures.forEach { it.get() }
}

/** indices의 각 i에 대해 task(i) 병렬 실행. */
internal fun turboForkJoinIndices(count: Int, task: (Int) -> Unit) {
    if (count <= 1) {
        for (i in 0 until count) task(i)
        return
    }
    turboForkJoinAll((0 until count).map { i -> { task(i) } })
}

internal fun turboSyncParamsData(target: List<TurboTensor>, source: List<TurboTensor>) {
    require(target.size == source.size) { "param count 불일치: ${target.size} vs ${source.size}" }
    for (i in target.indices) {
        val t = target[i]
        val s = source[i]
        require(t.numel == s.numel) { "param $i 크기 불일치: ${t.numel} vs ${s.numel}" }
        s.data.copyInto(t.data)
    }
}

internal fun turboAccumulateGrads(target: List<TurboTensor>, source: List<TurboTensor>) {
    require(target.size == source.size)
    for (i in target.indices) {
        val sg = source[i].grad ?: continue
        val tg = target[i].gradOrAlloc()
        for (j in tg.indices) tg[j] += sg[j]
    }
}

internal fun <T> turboDistributeRoundRobin(items: List<T>, n: Int): List<List<T>> {
    require(n > 0)
    val chunks = Array(n) { mutableListOf<T>() }
    for ((i, item) in items.withIndex()) chunks[i % n].add(item)
    return chunks.map { it.toList() }
}
