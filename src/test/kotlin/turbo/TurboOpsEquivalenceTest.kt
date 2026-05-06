package turbo

import turbo.ops.turboGelu
import turbo.ops.turboGeluBackward
import turbo.ops.turboSilu
import turbo.ops.turboSiluBackward
import turbo.ops.turboSoftmax
import turbo.ops.turboSoftmaxBackward
import turbo.ops.turboLayerNormForward
import turbo.ops.turboLayerNormBackward
import turbo.ops.turboCrossEntropyForward
import turbo.ops.turboCrossEntropyBackward
import turbo.ops.turboApplyRoPE
import turbo.ops.turboApplyRoPEBackward
import vec.Tensor as VecTensor
import vec.ops.gelu as vecGelu
import vec.ops.geluBackward as vecGeluBackward
import vec.ops.silu as vecSilu
import vec.ops.siluBackward as vecSiluBackward
import vec.ops.softmax as vecSoftmax
import vec.ops.softmaxBackward as vecSoftmaxBackward
import vec.ops.layerNormForward as vecLayerNormForward
import vec.ops.layerNormBackward as vecLayerNormBackward
import vec.ops.crossEntropyForward as vecCrossEntropyForward
import vec.ops.crossEntropyBackward as vecCrossEntropyBackward
import vec.ops.applyRoPE as vecApplyRoPE
import vec.ops.applyRoPEBackward as vecApplyRoPEBackward
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Phase 0 동등성: 모든 ops의 forward/backward가 vec와 수치 일치 (maxAbsDiff < 1e-5).
 * 학습 1초 룰을 위해 모든 입력 shape는 작게 유지.
 */
class TurboOpsEquivalenceTest {

    @Test
    fun geluMatches() {
        val rng = Random(1)
        val data = FloatArray(40) { rng.nextFloat() * 4f - 2f }
        val gy = FloatArray(40) { rng.nextFloat() - 0.5f }

        val tx = TurboTensor(intArrayOf(5, 8), data.copyOf())
        val ty = turboGelu(tx)
        val tdx = turboGeluBackward(tx, gy)

        val vx = VecTensor(intArrayOf(5, 8), data.copyOf())
        val vy = vecGelu(vx)
        val vdx = vecGeluBackward(vx, gy)

        assertCloseFloatArray(ty.data, vy.data, 1e-6f, "GELU forward")
        assertCloseFloatArray(tdx, vdx, 1e-6f, "GELU backward")
    }

    @Test
    fun siluMatches() {
        val rng = Random(2)
        val data = FloatArray(40) { rng.nextFloat() * 4f - 2f }
        val gy = FloatArray(40) { rng.nextFloat() - 0.5f }

        val tx = TurboTensor(intArrayOf(5, 8), data.copyOf())
        val vx = VecTensor(intArrayOf(5, 8), data.copyOf())

        assertCloseFloatArray(turboSilu(tx).data, vecSilu(vx).data, 1e-6f, "SiLU forward")
        assertCloseFloatArray(turboSiluBackward(tx, gy), vecSiluBackward(vx, gy), 1e-6f, "SiLU backward")
    }

    @Test
    fun softmaxMatches() {
        val rng = Random(3)
        val data = FloatArray(20) { rng.nextFloat() * 6f - 3f }
        val gy = FloatArray(20) { rng.nextFloat() - 0.5f }

        val tx = TurboTensor(intArrayOf(4, 5), data.copyOf())
        val vx = VecTensor(intArrayOf(4, 5), data.copyOf())

        val tsm = turboSoftmax(tx)
        val vsm = vecSoftmax(vx)
        assertCloseFloatArray(tsm.data, vsm.data, 1e-6f, "Softmax forward")
        assertCloseFloatArray(turboSoftmaxBackward(tsm, gy), vecSoftmaxBackward(vsm, gy), 1e-6f, "Softmax backward")
    }

    @Test
    fun layerNormMatches() {
        val rng = Random(4)
        val n = 4; val c = 8
        val data = FloatArray(n * c) { rng.nextFloat() - 0.5f }
        val gammaData = FloatArray(c) { 1.0f + 0.1f * (rng.nextFloat() - 0.5f) }
        val betaData = FloatArray(c) { 0.05f * (rng.nextFloat() - 0.5f) }
        val gy = FloatArray(n * c) { rng.nextFloat() - 0.5f }

        val tx = TurboTensor(intArrayOf(n, c), data.copyOf())
        val tg = TurboTensor(intArrayOf(c), gammaData.copyOf())
        val tb = TurboTensor(intArrayOf(c), betaData.copyOf())
        val (ty, tcache) = turboLayerNormForward(tx, tg, tb)
        val tdx = turboLayerNormBackward(tcache, tg, tb, gy.copyOf(), n, c)

        val vx = VecTensor(intArrayOf(n, c), data.copyOf())
        val vg = VecTensor(intArrayOf(c), gammaData.copyOf())
        val vb = VecTensor(intArrayOf(c), betaData.copyOf())
        val (vy, vcache) = vecLayerNormForward(vx, vg, vb)
        val vdx = vecLayerNormBackward(vcache, vg, vb, gy.copyOf(), n, c)

        assertCloseFloatArray(ty.data, vy.data, 1e-5f, "LayerNorm forward")
        assertCloseFloatArray(tdx, vdx, 1e-5f, "LayerNorm dx")
        assertCloseFloatArray(tg.grad!!, vg.grad!!, 1e-5f, "LayerNorm dgamma")
        assertCloseFloatArray(tb.grad!!, vb.grad!!, 1e-5f, "LayerNorm dbeta")
    }

    @Test
    fun crossEntropyMatches() {
        val rng = Random(5)
        val n = 4; val v = 6
        val data = FloatArray(n * v) { rng.nextFloat() * 4f - 2f }
        val targets = IntArray(n) { rng.nextInt(v) }

        val tx = TurboTensor(intArrayOf(n, v), data.copyOf())
        val tres = turboCrossEntropyForward(tx, targets)
        val tdx = turboCrossEntropyBackward(tx, targets, tres.softmax)

        val vx = VecTensor(intArrayOf(n, v), data.copyOf())
        val vres = vecCrossEntropyForward(vx, targets)
        val vdx = vecCrossEntropyBackward(vx, targets, vres.softmax)

        assertTrue(abs(tres.loss - vres.loss) < 1e-5f, "CE loss mismatch: ${tres.loss} vs ${vres.loss}")
        assertCloseFloatArray(tres.softmax.data, vres.softmax.data, 1e-6f, "CE softmax")
        assertCloseFloatArray(tdx.data, vdx.data, 1e-6f, "CE backward")
    }

    @Test
    fun ropeMatches() {
        val rng = Random(6)
        val t = 6; val h = 2; val d = 4
        val data = FloatArray(t * h * d) { rng.nextFloat() - 0.5f }

        val tx = TurboTensor(intArrayOf(t, h * d), data.copyOf())
        turboApplyRoPE(tx, h)

        val vx = VecTensor(intArrayOf(t, h * d), data.copyOf())
        vecApplyRoPE(vx, h)

        assertCloseFloatArray(tx.data, vx.data, 1e-6f, "RoPE forward")

        val gyData = FloatArray(t * h * d) { rng.nextFloat() - 0.5f }
        val tdx = TurboTensor(intArrayOf(t, h * d), gyData.copyOf())
        turboApplyRoPEBackward(tdx, h)
        val vdx = VecTensor(intArrayOf(t, h * d), gyData.copyOf())
        vecApplyRoPEBackward(vdx, h)
        assertCloseFloatArray(tdx.data, vdx.data, 1e-6f, "RoPE backward")
    }

    private fun assertCloseFloatArray(a: FloatArray, b: FloatArray, tol: Float, label: String) {
        require(a.size == b.size) { "$label: size mismatch ${a.size} vs ${b.size}" }
        var maxD = 0.0f
        for (i in a.indices) {
            val d = abs(a[i] - b[i])
            if (d > maxD) maxD = d
        }
        assertTrue(maxD <= tol, "$label maxDiff=$maxD > tol=$tol")
    }
}
