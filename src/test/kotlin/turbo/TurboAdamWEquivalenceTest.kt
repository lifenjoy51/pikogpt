package turbo

import vec.Tensor as VecTensor
import vec.VecAdamW
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Phase 2 — TurboAdamW SIMD fused 단일 패스 루프가 vec.VecAdamW (scalar 루프)와
 * 수치적으로 동일한 step 결과를 내는지 검증. 같은 hyperparameter + 같은 grad 시퀀스에서
 * 여러 step 누적 후 weight 차이가 1e-5 이내여야.
 */
class TurboAdamWEquivalenceTest {

    @Test
    fun adamWStepMatchesVec() {
        val rng = Random(42)
        val len = 64
        val weight = FloatArray(len) { rng.nextFloat() - 0.5f }

        val tParam = TurboTensor(intArrayOf(8, 8), weight.copyOf())
        val vParam = VecTensor(intArrayOf(8, 8), weight.copyOf())

        val tOpt = TurboAdamW(
            parameters = listOf(tParam),
            learningRate = 1e-3f,
            beta1 = 0.9f,
            beta2 = 0.999f,
            weightDecay = 0.01f,
            epsilon = 1e-8f,
        )
        val vOpt = VecAdamW(
            parameters = listOf(vParam),
            learningRate = 1e-3f,
            beta1 = 0.9f,
            beta2 = 0.999f,
            weightDecay = 0.01f,
            epsilon = 1e-8f,
        )

        // 5 step 누적
        for (step in 0 until 5) {
            val grad = FloatArray(len) { rng.nextFloat() - 0.5f }
            val tGrad = tParam.gradOrAlloc()
            val vGrad = vParam.gradOrAlloc()
            grad.copyInto(tGrad)
            grad.copyInto(vGrad)

            tOpt.step()
            vOpt.step()

            tParam.zeroGrad()
            vParam.zeroGrad()
        }

        var maxD = 0.0f
        for (i in 0 until len) {
            val d = abs(tParam.data[i] - vParam.data[i])
            if (d > maxD) maxD = d
        }
        assertTrue(maxD < 1e-5f, "AdamW step 결과 mismatch: maxDiff=$maxD")
    }

    @Test
    fun adamWWithVaryingGradMagnitudesStable() {
        val rng = Random(99)
        val len = 128
        val weight = FloatArray(len) { rng.nextFloat() - 0.5f }
        val tParam = TurboTensor(intArrayOf(len), weight.copyOf())
        val vParam = VecTensor(intArrayOf(len), weight.copyOf())
        val tOpt = TurboAdamW(listOf(tParam), 1e-3f, 0.9f, 0.95f, 0.0f, 1e-8f)  // wd=0
        val vOpt = VecAdamW(listOf(vParam), 1e-3f, 0.9f, 0.95f, 0.0f, 1e-8f)

        for (step in 0 until 10) {
            // 진폭 다양 — small/large grad mix
            val mag = if (step % 2 == 0) 1e-4f else 5.0f
            val grad = FloatArray(len) { mag * (rng.nextFloat() - 0.5f) }
            grad.copyInto(tParam.gradOrAlloc())
            grad.copyInto(vParam.gradOrAlloc())
            tOpt.step()
            vOpt.step()
            tParam.zeroGrad()
            vParam.zeroGrad()
        }

        var maxD = 0.0f
        for (i in 0 until len) {
            val d = abs(tParam.data[i] - vParam.data[i])
            if (d > maxD) maxD = d
        }
        assertTrue(maxD < 1e-5f, "AdamW varying-magnitude step 결과 mismatch: maxDiff=$maxD")
    }
}
