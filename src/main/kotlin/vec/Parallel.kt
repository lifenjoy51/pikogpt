package vec

/**
 * 데이터 병렬 학습에 필요한 작은 유틸리티.
 *
 * 벡터 `Trainer`는 한 optimizer step에서 `accum × batch` 개의 시퀀스를 forward/backward한다.
 * 각 시퀀스는 독립이라 workers에 나눠 담당시킬 수 있는데, 이때
 *
 *   1) 매 iter 시작에 worker 파라미터 데이터를 master에서 복사 (`syncParamsData`)
 *   2) 각 worker가 자기 param.grad에 backward 결과 누적 (worker별 독립 grad 버퍼)
 *   3) iter 끝에 worker grads의 합을 master.grad에 누적 (`accumulateGrads`)
 *
 * 식으로 PyTorch의 `DistributedDataParallel`과 같은 수식: 파라미터는 동일하게 유지되고
 * gradient만 병렬로 만들어 합친다.
 */

/** 각 파라미터의 `data` FloatArray를 source → target으로 복사. 쉐이프 일치 가정. */
internal fun syncParamsData(target: List<Tensor>, source: List<Tensor>) {
    require(target.size == source.size) { "param count 불일치: ${target.size} vs ${source.size}" }
    for (i in target.indices) {
        val t = target[i]
        val s = source[i]
        require(t.numel == s.numel) { "param $i 크기 불일치: ${t.numel} vs ${s.numel}" }
        s.data.copyInto(t.data)
    }
}

/** source 각 파라미터의 grad를 target의 grad에 원소별로 더한다. source grad가 null이면 건너뜀. */
internal fun accumulateGrads(target: List<Tensor>, source: List<Tensor>) {
    require(target.size == source.size)
    for (i in target.indices) {
        val sg = source[i].grad ?: continue
        val tg = target[i].gradOrAlloc()
        for (j in tg.indices) tg[j] += sg[j]
    }
}

/**
 * 주어진 리스트를 `n`개 청크로 분배 (round-robin). 각 청크 길이 차이가 최대 1.
 * 예: `distributeRoundRobin([a,b,c,d,e], 3)` → `[[a,d], [b,e], [c]]`.
 */
internal fun <T> distributeRoundRobin(items: List<T>, n: Int): List<List<T>> {
    require(n > 0)
    val chunks = Array(n) { mutableListOf<T>() }
    for ((i, item) in items.withIndex()) chunks[i % n].add(item)
    return chunks.map { it.toList() }
}
