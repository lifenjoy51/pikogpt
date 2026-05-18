package mps

import kotlin.math.abs

/**
 * P0.3 — best loss tracking + smoothing + early stopping.
 *
 * TurboTrainer의 동일 로직(라인 252~303)을 standalone으로 분리해서 단위 test 가능하게 함.
 *
 *   smoothed = 최근 [smoothingWindow] eval의 평균 (단발 outlier가 best를 빼앗지 않게)
 *   isBest    = smoothed < bestLoss
 *   patience  = isBest 갱신 후 N회 연속 best 갱신 없으면 stop (earlyStopPatience > 0일 때)
 *   plateau   = 최근 [plateauWindow] eval smoothed의 |first - last| < initialSmoothed × plateauRelTol
 *               + patience ≥ plateauMinPatience 동시 만족 시 stop
 */
class MpsBestLossTracker(
    val smoothingWindow: Int = 5,
    val earlyStopPatience: Int = 0,
    val plateauWindow: Int = 10,
    val plateauRelTol: Double = 0.001,
    val plateauMinPatience: Int = 5,
    initialBestLoss: Double = Double.POSITIVE_INFINITY,
) {
    private val recentSmoothing = mutableListOf<Double>()
    private val recentPlateau = mutableListOf<Double>()
    private var initialSmoothed: Double = 0.0

    var bestLoss: Double = initialBestLoss
        private set
    var patienceCounter: Int = 0
        private set
    var lastSmoothedLoss: Double = Double.NaN
        private set

    fun update(valLoss: Double): UpdateResult {
        recentSmoothing.add(valLoss)
        if (recentSmoothing.size > smoothingWindow) recentSmoothing.removeAt(0)
        val smoothed = recentSmoothing.average()
        lastSmoothedLoss = smoothed

        val isBest = smoothed < bestLoss
        if (isBest) {
            bestLoss = smoothed
            patienceCounter = 0
        } else {
            patienceCounter += 1
        }

        if (initialSmoothed == 0.0) initialSmoothed = smoothed
        recentPlateau.add(smoothed)
        if (recentPlateau.size > plateauWindow) recentPlateau.removeAt(0)

        val plateauDelta = if (recentPlateau.size >= plateauWindow) {
            abs(recentPlateau.first() - recentPlateau.last())
        } else Double.MAX_VALUE
        val plateauThreshold = initialSmoothed * plateauRelTol

        val shouldStopByPatience = earlyStopPatience > 0 && patienceCounter >= earlyStopPatience
        val shouldStopByPlateau = earlyStopPatience > 0 &&
            plateauDelta < plateauThreshold &&
            patienceCounter >= plateauMinPatience

        return UpdateResult(
            smoothedLoss = smoothed,
            isBest = isBest,
            shouldStopByPatience = shouldStopByPatience,
            shouldStopByPlateau = shouldStopByPlateau,
        )
    }

    data class UpdateResult(
        val smoothedLoss: Double,
        val isBest: Boolean,
        val shouldStopByPatience: Boolean,
        val shouldStopByPlateau: Boolean,
    )
}
