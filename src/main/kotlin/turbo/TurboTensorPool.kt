package turbo

/**
 * Per-thread activation FloatArray 풀. Phase 0은 **OFF 모드 스켈레톤** — 매 acquire가
 * 새 FloatArray를 할당하고 release는 no-op이라 vec과 동일한 GC 동작.
 *
 * Phase 2에서 LIFO scope 풀로 활성화하고 hot path에서 재사용해 GC 부담을 ~0에 가깝게 만든다.
 *
 *   - acquire(size) → FloatArray   (scope 동안 빌림)
 *   - release(buf)                 (scope 종료 시 일괄 회수, 또는 명시적 반환)
 *   - beginScope() / endScope()    (forward+backward 한 iter scope)
 *
 * 사용 패턴 (Phase 2 이후):
 *   pool.beginScope()
 *   try { /* forward + backward */ } finally { pool.endScope() }
 */
class TurboTensorPool(val enabled: Boolean = false) {

    fun beginScope() { /* Phase 0 no-op */ }
    fun endScope() { /* Phase 0 no-op */ }

    /** 현재 모드에서 size 크기 FloatArray를 빌려준다. Phase 0은 항상 새로 할당. */
    fun acquire(size: Int): FloatArray = FloatArray(size)

    /** 빌린 배열을 반환. Phase 0은 no-op (GC가 회수). */
    @Suppress("UNUSED_PARAMETER")
    fun release(buf: FloatArray) { /* Phase 0 no-op */ }

    companion object {
        /** Per-thread 풀 핸들. Phase 0은 disabled. */
        private val threadLocal: ThreadLocal<TurboTensorPool> = ThreadLocal.withInitial {
            TurboTensorPool(enabled = false)
        }
        fun current(): TurboTensorPool = threadLocal.get()
    }
}
