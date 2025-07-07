package train

import kotlinx.serialization.Serializable

@Serializable
data class AttentionState(
    val qProj: LinearState,
    val kProj: LinearState,
    val vProj: LinearState,
    val outProj: LinearState
)

@Serializable
data class BlockState(
    val ln1: LayerNormState,
    val attn: AttentionState,
    val ln2: LayerNormState,
    val ffn: FeedForwardState
)

@Serializable
data class FeedForwardState(
    val cFc: LinearState,
    val cProj: LinearState
)

@Serializable
data class ModelState(
    val tokenEmbedding: List<List<Double>>,
    val positionEmbedding: List<List<Double>>,
    val blocks: List<BlockState>,
    val lmHead: LinearState
)

@Serializable
data class LayerNormState(
    val weight: List<Double>,
    val bias: List<Double>?
)

@Serializable
data class LinearState(
    val weight: List<List<Double>>,
    val bias: List<Double>?
)

@Serializable
data class OptimizerState(
    val iteration: Int,
    val m: Map<String, List<Double>>, // 1차 모멘트
    val v: Map<String, List<Double>>  // 2차 모멘트
)
