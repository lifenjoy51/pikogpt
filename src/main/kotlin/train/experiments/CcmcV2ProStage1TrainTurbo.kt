package train.experiments

import turbo.TurboTrainConfig

/**
 * **CCMC v2-pro Stage 1 (binding)** — sensory + category + multi_role + contrast가 섞인
 * lemma-grounded 짧은 문장 묶음으로 의미 binding을 1차 주입. scratch 출발.
 *
 * 데이터: `data/ccmc-v2-pro/stage1` (`train.txt` + `val.txt` + `meta.json` — shared vocab으로 인코딩됨)
 *   - 형식: `<|bos|><|sep|><sentences>...<|eos|>` (한 줄 = 1 record, literal \n은 <|sep|>로 치환됨)
 *   - 분할 기준: 토큰 수 9:1 (record 단위 shuffle 후 누적 토큰으로 컷, seed=42)
 *   - 토큰 수: train ~974k, val ~108k.
 *
 * 모델: layers 8 · dim 96 · heads 3 · block 64 · tied + SwiGLU + RoPE (~0.86M~0.9M)
 *   TwoStageBase 6L 아키텍처를 깊이만 +2 확장. blockSize/MLP 활성/positional 모두 검증된 구성 유지.
 *
 * 학습 설정:
 *   - effective batch 2048 token/step (block 32 × batch 2 × gradAccum 32)
 *   - maxIters 10000 → 10000 × 2048 ≈ 20M token 노출 ≈ 21 epoch over Stage1 ~974k tokens.
 *   - blockSize 32 + chunkAnchoredSampling: 한 chunk anchor 평균 21.5회 학습 (random offset 대비 26×).
 *   - LR 3e-4 (cosine decay 0.95), warmup 3%, label smoothing 0.05 — TwoStageBase와 동일.
 *
 * 산출 ckpt 경로: `model/stage1/vec/<paramCount>/v00XX/` (datasetName = "stage1")
 *   Stage 2 finetune에서 `pretrainCheckpointDir`로 이 경로를 가리킨다.
 */
fun main(args: Array<String>) {
    val resume = args.any { it.equals("resume", ignoreCase = true) }
    val maxItersOverride = args.firstOrNull { !it.equals("resume", ignoreCase = true) }?.toIntOrNull()

    val config = TurboTrainConfig(
        dataPath = "data/ccmc-v2-pro/stage1",
        modelDir = "model",
        samplePrompts = listOf(
            // 각 lemma별 학습 데이터에서 가장 빈번한 prefix 사용 — sentence-mid in-distribution.
            // 명사: "the X" 또는 "a X" 중 빈도 top, 동사: "to X", 형용사: "a X" 빈도 top, 그 외: lemma만.
            // 끝 공백 1개로 다음 단어가 시작되는 위치 명시.
            "the cat ",       // 357
            "the water ",     // 521
            "the tree ",      // 252
            "to run ",        // 64
            "to eat ",        // 181
            "a happy ",       // 182
            "a big ",         // 2135
            "above ",         // article 부적절, 앞 공백 빼고 통일
            "quickly ",       // article 부적절
            "and ",           // 단독
        ),
        gradientAccumulationSteps = 32,
        batchSize = 2,
        blockSize = 32,
        numberOfLayers = 8,
        numberOfHeads = 3,
        embeddingDimension = 96,
        dropout = 0.05f,
        bias = true,
        tieWeights = true,
        mlpActivation = "swiglu",
        positionEncoding = "rope",
        learningRate = 3e-4f,
        weightDecay = 0.05f,
        labelSmoothing = 0.05f,
        gradClip = 1.0f,
        beta1 = 0.9f,
        beta2 = 0.95f,
        maxIters = maxItersOverride ?: 10000,
        warmupRatio = 0.03f,
        learningRateDecayRatio = 0.95f,
        minimumLearningRate = 3e-5f,
        decayLr = true,
        evalIntervalRatio = 0.02f,
        evalIters = 100,
        logInterval = 100,
        alwaysSaveCheckpoint = true,
        recordAwareSampling = false,
        chunkAnchoredSampling = true,
        initFrom = if (resume) "resume" else "scratch",
    )

    turbo.TurboTrainer(config).train()
}
