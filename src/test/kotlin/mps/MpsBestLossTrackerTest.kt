package mps

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class MpsBestLossTrackerTest {

    @Test
    fun bestLossDecreasesOverTime() {
        val tr = MpsBestLossTracker(smoothingWindow = 1)
        tr.update(3.0).also { assertTrue(it.isBest) }
        tr.update(2.0).also { assertTrue(it.isBest) }
        tr.update(2.5).also { assertFalse(it.isBest) }
        tr.update(1.5).also { assertTrue(it.isBest) }
        assertEquals(1.5, tr.bestLoss, 1e-9)
    }

    @Test
    fun smoothingPreventsOutlierFromUpdatingBest() {
        val tr = MpsBestLossTracker(smoothingWindow = 3)
        tr.update(3.0)
        tr.update(3.0)
        tr.update(3.0)
        // 단발 outlier 1.0 — smoothed = (3+3+1)/3 ≈ 2.33 → 3.0보다 작아 best 갱신
        val r1 = tr.update(1.0)
        assertTrue(r1.isBest)
        // 그러나 다음 step에서 다시 3.0이 들어오면 smoothed (3+1+3)/3 = 2.33 동일 → not best
        val r2 = tr.update(3.0)
        assertFalse(r2.isBest)
    }

    @Test
    fun earlyStopPatienceFiresAfterNNonImprovements() {
        val tr = MpsBestLossTracker(smoothingWindow = 1, earlyStopPatience = 3)
        tr.update(2.0)  // best
        val r1 = tr.update(2.1); assertFalse(r1.shouldStopByPatience)
        val r2 = tr.update(2.1); assertFalse(r2.shouldStopByPatience)
        val r3 = tr.update(2.1); assertTrue(r3.shouldStopByPatience)
    }

    @Test
    fun patienceResetsOnBestUpdate() {
        val tr = MpsBestLossTracker(smoothingWindow = 1, earlyStopPatience = 3)
        tr.update(2.0)
        tr.update(2.5)
        tr.update(2.5)
        assertEquals(2, tr.patienceCounter)
        tr.update(1.9)
        assertEquals(0, tr.patienceCounter)
    }

    @Test
    fun plateauNotFiredBeforeFullWindow() {
        val tr = MpsBestLossTracker(
            smoothingWindow = 1, earlyStopPatience = 100,
            plateauWindow = 10, plateauRelTol = 0.001, plateauMinPatience = 5,
        )
        // 너무 짧은 history는 plateau로 보지 않음
        for (i in 0 until 5) tr.update(2.0)
        for (i in 0 until 4) {
            val r = tr.update(2.0)
            assertFalse(r.shouldStopByPlateau, "i=$i")
        }
    }
}
