package mps

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MpsLrScheduleTest {

    @Test
    fun warmupIsLinearFromZero() {
        val lr0 = MpsLrSchedule.computeLr(iter = 0, warmupIters = 10, decayIters = 100,
            baseLr = 1e-3f, minLr = 1e-5f)
        val lr5 = MpsLrSchedule.computeLr(iter = 4, warmupIters = 10, decayIters = 100,
            baseLr = 1e-3f, minLr = 1e-5f)
        val lr10 = MpsLrSchedule.computeLr(iter = 9, warmupIters = 10, decayIters = 100,
            baseLr = 1e-3f, minLr = 1e-5f)
        // (iter+1)/warmup * base  → 0.1, 0.5, 1.0 * base
        assertEquals(1e-4f, lr0, 1e-9f)
        assertEquals(5e-4f, lr5, 1e-9f)
        assertEquals(1e-3f, lr10, 1e-9f)
    }

    @Test
    fun decayCosineHitsHalfAtMidpoint() {
        // iter == warmup + (decay - warmup)/2 → coefficient = 0.5*(1+cos(π/2)) = 0.5
        val mid = 10 + (100 - 10) / 2
        val lr = MpsLrSchedule.computeLr(mid, warmupIters = 10, decayIters = 100,
            baseLr = 1e-3f, minLr = 1e-5f)
        val expected = 1e-5f + 0.5f * (1e-3f - 1e-5f)
        assertTrue(abs(lr - expected) < 1e-7f, "mid lr=$lr expected=$expected")
    }

    @Test
    fun afterDecayBecomesMinLr() {
        val lr = MpsLrSchedule.computeLr(iter = 500, warmupIters = 10, decayIters = 100,
            baseLr = 1e-3f, minLr = 1e-5f)
        assertEquals(1e-5f, lr, 1e-9f)
    }

    @Test
    fun decayDisabledReturnsBase() {
        val lr = MpsLrSchedule.computeLr(iter = 500, warmupIters = 10, decayIters = 100,
            baseLr = 3e-4f, minLr = 1e-5f, decayEnabled = false)
        assertEquals(3e-4f, lr, 1e-9f)
    }
}
