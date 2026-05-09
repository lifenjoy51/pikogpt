# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PikoGPT is a Kotlin port of nanoGPT and micrograd by Andrej Karpathy. This is an educational project that implements a GPT model with automatic differentiation from scratch in Kotlin, without external ML libraries.

## Build and Development Commands

```bash
# Build
./gradlew build
./gradlew clean

# Run all tests (fast: 전체 1초 이내가 목표. training/sampling 배관 smoke test도 포함)
./gradlew test
./gradlew test --tests "ValueTest"
./gradlew test --tests "pipeline.FullPipelineTest"  # 합성 데이터 기반 end-to-end 배관 검증

# 실행 스크립트 — 전부 main 소스셋 + JavaExec 기반.
./gradlew runMain                # MainKt
./gradlew runBpe                 # BpePrep — BPE 학습 + 인코딩 (-Xmx12g)
./gradlew runAlphabetPrep        # data/alphabet/az.txt → train.bin/val.bin/meta.json
./gradlew runEncodeWithExistingMeta  # 공유 meta.json으로 다른 디렉터리 인코딩
./gradlew runSplitByTokenRatio   # record-per-line 입력을 토큰 수 기준으로 train/val 분할
./gradlew runBPETest             # BPETestKt
./gradlew runTrainer             # TrainerMain (-Xmx8g)
./gradlew runMiniTrainer         # Scalar 백엔드 quickstart 학습 (data/alphabet, ~10분)
./gradlew runSampler             # Scalar quickstart 샘플링 (인자 없으면 model/alphabet/main 자동)
./gradlew runSampler --args="model/alphabet/main/v0001"  # 명시 ckpt
./gradlew runAnalyzeTokens       # 토큰 분포 분석
./gradlew runDebugBPE            # BPE 디버그

# Scalar end-to-end (한 흐름 가이드: docs/scalar-quickstart.md)
./gradlew runAlphabetPrep && ./gradlew runMiniTrainer && ./gradlew runSampler

# Turbo 백엔드 학습/샘플링 (1M~3M 파라미터)
./gradlew runTinyHelenTrainTurbo            # 혼합 코퍼스(book+textbook+wiki+conversation) 10k
./gradlew runTinyHelenTrainTextbookTurbo    # textbook-only 6k
./gradlew runTinyHelenSampleTurbo           # 최신 turbo ckpt 자동 샘플링
./gradlew runChatTurbo                      # turbo ckpt 인터랙티브 REPL (ckpt 인자 필요)
./gradlew runSamplePromptsFromFile \
          --args="<ckpt-dir> <prompts.txt>"  # 커스텀 프롬프트 파일 샘플링
./gradlew runInferenceApi                   # Turbo HTTP 추론 API (ckpt + port 인자)

# Resume (turbo)
./gradlew runTinyHelenTrainTurbo --args="resume"           # 최대 iter ckpt에서 이어하기
./gradlew runTinyHelenTrainTurbo --args="20000 resume"     # maxIters 늘려 이어하기

# 벤치마크
./gradlew runTurboBench          # MatMul / AdamW 마이크로벤치
./gradlew runBench5MTurbo        # 5M params 500 iter
./gradlew runBench10MTurbo       # 10M params 250 iter
```

NOTE: `application` 플러그인을 쓰지 않으므로 `./gradlew run`은 없다. 위 `runXxx` 태스크를 사용한다. 역할 분리 원칙:
- **`src/test/kotlin/`** — `@Test`만. 전체 테스트 스윗이 1초 내로 끝나도록 유지.
- **`src/main/kotlin/`** — 프로덕션 코드 + 실행 스크립트(`fun main()`). 오래 걸려도 됨.

`train/experiments/`에는 32개의 실험 진입점(`ConvMix*`, `TinyHelen*`, `ThreeStage*`, `TwoStage*`, `Ccmc*`, `Bench*`, `Dialogues*`)이 있고, 각각 `runXxx` 태스크로 노출된다. 전체 목록은 `build.gradle.kts` 참조.

## Toolchain

- Kotlin JVM `1.9.0` (+ `kotlin("plugin.serialization") 1.9.0`)
- `org.jetbrains.kotlinx:kotlinx-serialization-json:1.5.0`
- `org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3`
- Ktor 2.3.12 (server-core, server-netty, content-negotiation, serialization-kotlinx-json) — `runInferenceApi` 전용
- Tests: `kotlin("test")` on JUnit Platform
- **JDK 21** — turbo 백엔드의 `jdk.incubator.vector` 모듈 사용 (`--add-modules=jdk.incubator.vector`). Kotlin 1.9.0이 jvmTarget=21을 지원하지 않아 컴파일 target은 17, 런타임은 JDK 21.

## Architecture

The codebase has **two parallel autodiff backends** with deliberately different priorities:

- **Scalar (`Value.kt` + `grad/` + `gpt/` + `train/Scalar*` + `sample/ScalarSampler.kt`)** — 교육용 레퍼런스.
  모든 연산이 `Value` 스칼라 객체 그래프 위에서 일어나고, `backward()`가 위상정렬로 체인 룰을
  재생한다. micrograd의 직계 포팅. 학습이 가장 읽기 쉽다는 것이 존재 이유. 단점: 71K 파라미터
  모델이 iter당 16초. 실험용으로는 느림.
- **Turbo (`turbo/`)** — 실전 성능 + SIMD 가속.
  `TurboTensor` = `FloatArray + shape`. 연산과 backward는 **autograd 없이 레이어마다 명시적**으로 기록
  (layer.forward / layer.backward를 나란히 읽으면 chain rule이 수식 그대로 드러남). JDK 21
  `jdk.incubator.vector` API 기반 SIMD + ForkJoinPool 병렬화 + KV cache (추론 5~10×).
  RMSNorm/SwiGLU/RoPE/GQA/qk-norm/fused QKV/z-loss 옵션 모두 지원. 1M 모델 iter당 ~2.6초.

두 백엔드는 **병행**한다 (서로를 대체하지 않음). 선택은 실행 시점에 진입점으로:
`runMiniTrainer`(scalar) vs `runTinyHelenTrainTurbo`(turbo) 등.

### 백엔드 명명 규칙

두 백엔드가 같은 패키지 트리에 공존하므로 **모든 동명 클래스에 `Turbo`/`Scalar` 접두사**를 붙여 코드만 봐도 어느 백엔드인지 분명하게 한다. 패키지 명시 없이 클래스명만으로 구분 가능. 예: `TurboTrainer` vs `ScalarTrainer`, `TurboSampler` vs `ScalarSampler`, `TurboAdamW` vs `ScalarAdamW`, `TurboPikoGPT` vs `ScalarPikoGPT`. micrograd 출처 클래스는 `Micrograd*` 접두사. `TurboTensor`(turbo)와 `Value`(scalar)는 충돌이 없어 그대로.

### 공유 정의

- `grad/` — micrograd-style scalar MLP (`MicrogradNeuron` → `MicrogradLayer` → `MicrogradMLP` → `MicrogradLossCalculator`). 스칼라 autodiff의 가장 단순한 예. `MlpTest` / `LossTest`로 autodiff 자체의 건강성을 검증.

### Core components

1. **Autodiff / utilities (root + util package)**
   - `Value.kt` — scalar autodiff; overloads `+ - * /`, `pow`, `ReLU`, `GELU`, `sigmoid`, and implements `backward()` via a reverse-topological traversal.
   - `RandomGaussian.kt` — standard normal sampler for weight init.
   - `util/FloatExtensions.kt` — Float용 `sumOf` 확장 (stdlib는 Double/Int/Long만 지원).

2. **micrograd-style MLP (`src/main/kotlin/grad/`)**
   - `MicrogradNeuron.kt`, `MicrogradLayer.kt`, `MicrogradMLP.kt`, `MicrogradLossCalculator.kt`
   - 클래스명에 `Micrograd` 접두사를 붙여 GPT의 `ScalarFeedForward`(transformer FFN)와 혼동 차단.

3. **GPT model (`src/main/kotlin/gpt/`)** — 스칼라 백엔드
   - `ScalarPikoGPT.kt` — top-level model (token + position embeddings, stacked `ScalarTransformerBlock`s, lm head).
   - `GPTConfig.kt` — layers, heads, embedding size, block size, dropout. 두 백엔드 공유. (long-name canonical: `embeddingDimension`, `numberOfLayers`, `maxSequenceLength`, ...)
   - `ScalarTransformerBlock.kt` — LN → attention → residual → LN → FFN → residual.
   - `ScalarCausalSelfAttention.kt` — causal multi-head attention (Q/K/V projections + masking). `forward()`가 5단계 헬퍼로 분해되어 수식과 1:1 대응.
   - `ScalarFeedForward.kt` — Transformer feed-forward (expand → GELU → contract).
   - `ScalarLayerNorm.kt`, `ScalarDropout.kt`, `ScalarLinear.kt`, `ScalarEmbeddingTable.kt`
   - `Matrix.kt` — Value 2D 행렬. 스칼라 백엔드의 유일한 텐서 표현. `mapRows`/`zipWith` + 의미적 helper(`lastRow`, `softmaxRows`, `argMaxRows`).

4. **Training (`src/main/kotlin/train/`)**
   - `ScalarTrainer.kt` — 스칼라 백엔드 학습 루프: LR schedule, grad clipping, eval, checkpointing.
   - `ScalarAdamW.kt` — 스칼라 백엔드 옵티마이저 (`Value` 단위).
   - `DataLoader.kt` — minibatch sampling over `train.bin` / `val.bin`. 두 백엔드 공유.
   - `TrainConfig.kt` — hyperparameters + data/model paths.
   - `ScalarCheckpoint.kt` — serializable wrapper (iteration, best loss, model args, optimizer state).
   - `States.kt` — per-layer serializable state DTOs used by `ScalarCheckpoint`.
   - `MiniTrainerMain.kt` — Scalar quickstart 진입점 (data/alphabet, ~10분).
   - `TrainerMain.kt` — 사용자 정의 학습용 참조 템플릿.
   - `experiments/` — turbo 백엔드용 실험 진입점들 (32개): `ConvMix*TrainTurbo.kt`, `TinyHelenTrain*Turbo.kt`, `ThreeStage*V4/V5*Turbo.kt`, `TwoStage*Turbo.kt`, `CcmcV*Turbo.kt`, `Dialogues*Turbo.kt`, `Bench*Turbo.kt` (SwiGLU/RoPE/GQA 변형 포함). 코어 학습 로직과 분리.

5. **Sampling (`src/main/kotlin/sample/`)**
   - `ScalarSampler.kt` — 스칼라 백엔드. 체크포인트 로드 + 텍스트 생성 (temperature / top-k).
   - `SamplerMain.kt` — Scalar quickstart 샘플링 진입점 (인자 없으면 자동 검색).
   - `SampleConfig.kt` — sampling parameters (long-name canonical: `modelDirectoryPath`, `numberOfSamples`, ...).
   - `TinyHelenSample.kt` — 스칼라 ckpt 자동 검색 샘플링.
   - `TinyHelenSampleTurbo.kt` — turbo ckpt 자동 검색 샘플링.
   - `ChatTurbo.kt` — turbo ckpt 인터랙티브 REPL.
   - `SamplePromptsFromFile.kt`, `SampleV4MergedSpaceSep3M.kt` — turbo ckpt + 프롬프트 파일 샘플링.

6. **Data processing (`src/main/kotlin/data/`)**
   - `CharBPE.kt` — char-level BPE train + encode/decode. 플래그: `lowercase`, `useWordPreTokenize` (GPT-2 스타일 regex 사전 분할), `standardBpeScoring`(빈도 기준 merge), `splitSpaceAsToken`(공백을 단일 토큰으로 고정 → 알파벳만 BPE merge), `verbose`. 학습된 상태를 `getMerges()`로 내보내고 `restore(stoi, merges)`로 복원 — Sampler가 학습과 **정확히 같은 토큰화를 재생**.
   - `BpePrep.kt` — 일반 BPE 전처리 파이프라인. `main(args)` CLI로 데이터 경로 지정. meta.json에 `merges` + 플래그까지 저장.
   - `AlphabetPrep.kt` — Scalar quickstart용 character-level 전처리.
   - `EncodeWithExistingMeta.kt` — 학습 끝난 meta.json을 다른 디렉터리에 재사용해 인코딩.
   - `SplitByTokenRatio.kt` — record-per-line 입력을 토큰 수 기준으로 train/val 분할.
   - `CcmcV4TinyStoriesPrep.kt`, `CcmcV4MergedPrep.kt`, `CcmcV4MergedSpaceSepPrep.kt`, `CcmcV5QaPrep.kt` — CCMC 코퍼스 전처리 진입점.
   - `MetaInfo.kt` — vocab metadata DTO. `merges`, `lowercase`, `useWordPreTokenize`, `specialTokens` 포함.
   - `AnalyzeTokensMain.kt`, `DebugBPEMain.kt` — 토큰 분포/BPE 디버깅 도구.

7. **Turbo backend (`src/main/kotlin/turbo/`)**
   - `TurboTensor.kt` — `shape: IntArray`, `data: FloatArray`, lazy `grad: FloatArray?`. 계산 그래프 없음.
   - `TurboTensorPool.kt` — forward pass 임시 텐서 재사용 풀 (메모리 할당 압력 완화).
   - `TurboSimdMath.kt` — JDK 21 Vector API helper (`SPECIES_PREFERRED`, `dot`, `fmaScalar`, `addArrays`).
   - `ops/` — 원자 연산. 각 파일에 forward + backward + 수식 주석. 거의 모두 SIMD化:
     `TurboMatMul.kt`, `TurboSoftmax.kt`, `TurboGELU.kt`, `TurboSiLU.kt`(SwiGLU용), `TurboLayerNormOp.kt`, `TurboRMSNormOp.kt`, `TurboCrossEntropy.kt`, `TurboRoPE.kt`, `TurboRoPESingle.kt`(KV cache용).
   - `layer/` — 모두 `Turbo` 접두사. `forward(x)` / `backward(gy)` 노출:
     `TurboLinear`, `TurboLayerNorm`, `TurboRMSNorm`, `TurboNorm`(sealed interface), `TurboDropout`, `TurboMLP`(GELU/SwiGLU 분기), `TurboSelfAttention`(multi-head causal + RoPE/GQA/qk-norm/fused QKV), `TurboTransformerBlock`(pre-LN + grad checkpointing 토글), `TurboEmbeddingTable`, `TurboPikoGPT`(RoPE 모드면 position embedding 제외).
   - `TurboAdamW.kt`, `TurboTrainer.kt`, `TurboSampler.kt`, `TurboCheckpoint.kt`, `TurboKVCache.kt`(추론 가속), `TurboModelConfig.kt`(GQA/qk-norm/fused QKV/z-loss 옵션) — 파라미터별 FloatArray 기반의 옵티마이저 + 학습 루프 + 샘플러.
   - `TurboParallel.kt` — ForkJoinPool 기반 데이터 병렬 학습. worker = `cpuCount × 2/3`.
   - `bench/TurboMicroBench.kt` — MatMul shape별 절대 시간 측정 (회귀 추적용).

8. **Inference HTTP API (`src/main/kotlin/server/`)**
   - `InferenceApiMain.kt` — Ktor + Netty 기반 경량 HTTP 추론 서버. Turbo ckpt와 port를 인자로 받아 토큰 단위 generate 엔드포인트 제공. 자세한 사양은 `docs/inference-api.md`.

### Tests (`src/test/kotlin/`)

- `pipeline/FullPipelineTest.kt` — 합성 데이터로 학습 → 체크포인트 → 샘플링 배관 회귀 보호.
- `turbo/` — 옵션별 회귀: `TurboGqaTest`, `TurboFusedQkvTest`, `TurboQkNormTest`, `TurboRMSNormTest`, `TurboZLossTest`, `TurboKVCacheTest`.
- 그 외 `ValueTest`, `ValueNoGradTest` 등 autodiff/모델/데이터 단위 테스트. 전체 스윗이 1초 안에 끝나는 것이 원칙.

### Design patterns

- **Pure Kotlin, no ML libs.** Everything (autodiff, layers, optimizer, tokenizer) is implemented in-repo.
- **Scalar autodiff graph.** `Value` builds a dynamic computation graph per forward pass; `backward()` walks it in reverse-topological order.
- **Explicit layer backward (turbo).** autograd 없이 layer마다 forward/backward를 직접 짜서 SIMD에 맞춤.
- **Serialization via kotlinx.serialization.** Configs, checkpoints, and per-layer `States.kt` DTOs are `@Serializable`; model weights go to a separate `.bin`.
- **Execution convention.** 모든 실행 스크립트는 `src/main/kotlin/`의 `fun main()`이며 `build.gradle.kts`의 `runXxx` `JavaExec` 태스크로 실행한다. 통합 CLI는 없다. 테스트 디렉토리에는 `@Test`만 둔다.

## Training Workflow

1. **Data preparation** — `BpePrep.kt` (또는 `AlphabetPrep.kt`, `Ccmc*Prep.kt`)이 텍스트를 `data/[dataset]/train.bin` + `val.bin` + `meta.json`으로 변환. `./gradlew runBpe` / `runAlphabetPrep` 등.
2. **Training** — Scalar 백엔드 quickstart는 `./gradlew runMiniTrainer` (data/alphabet, ~10분). Turbo 백엔드는 `runTinyHelenTrainTurbo` 등. 체크포인트는 best 갱신 또는 매 평가마다 저장.
3. **Checkpoint layout** — 두 백엔드 모두 `${config.modelDir}/${datasetName}/${config.expName}/v0001/` (4자리 zero-pad 버전) 경로에 `checkpoint.json`, `model_weights.bin`, `meta.json`을 쓴다. Turbo는 `optimizer_state.bin`도 추가. 매 저장마다 v 번호 +1. `expName` default `"main"`, 같은 datasetName 공유하는 진입점만 unique 값 명시.
4. **Sampling** — Scalar는 `./gradlew runSampler` (인자 없으면 자동 검색). Turbo는 `runTinyHelenSampleTurbo` / `runChatTurbo` / `runSamplePromptsFromFile`. 자세한 가이드: `docs/scalar-quickstart.md`.

## Data Layout

```
data/
├── alphabet/                 # Scalar quickstart 데이터 (a-z 영문 패턴 텍스트, ~6KB)
│   ├── az.txt                # commit된 원본
│   ├── train.bin / val.bin   # runAlphabetPrep 산출물
│   └── meta.json
├── [dataset]/                # 예: tinyhelen, three-stage-v4, ccmc-v4-merged-spacesep
│   ├── stories.txt           # 단일 입력 (fallback: 90:10 cut)
│   ├── train.txt / val.txt   # 분리 입력 (있으면 우선, vocab은 train에서만 학습 — leakage 차단)
│   ├── meta.json             # vocab + token<->id maps + BPE merges + 플래그
│   ├── train.bin             # tokenized training set
│   ├── val.bin               # tokenized validation set
│   └── unique_words.txt      # optional vocab dump

model/                        # 모든 체크포인트 루트 (gitignored)
└── [datasetName]/            # config.dataPath 마지막 segment (예: alphabet, tinyhelen, stage2)
    └── [expName]/            # 사람이 정한 실험 이름 (default "main"; 예: bench5m, v4, m773-swiglu)
        └── v0001/            # 4자리 zero-pad 버전 (매 저장마다 +1). Scalar/Turbo 동일 schema.
            ├── checkpoint.json   # iteration, best loss, model args
            ├── meta.json         # data 디렉터리에서 복사
            ├── model_weights.bin # serialized weights (big-endian float32)
            └── optimizer_state.bin # Turbo만. AdamW timeStep + 모멘트 (resume용)
```

## Key Files to Understand

- `Value.kt` — autodiff foundation.
- `gpt/Matrix.kt` — shape-safe matrix layer underpinning the GPT stack.
- `gpt/ScalarPikoGPT.kt`, `gpt/GPTConfig.kt` — 스칼라 백엔드 모델 아키텍처 + config.
- `gpt/ScalarTransformerBlock.kt`, `gpt/ScalarCausalSelfAttention.kt`, `gpt/ScalarFeedForward.kt` — block internals (scalar).
- `turbo/layer/TurboPikoGPT.kt`, `turbo/TurboTrainer.kt` — turbo 백엔드 모델 + 학습 루프 (실전 성능, SIMD + KV cache + SwiGLU/RoPE/GQA 지원).
- `train/ScalarTrainer.kt`, `train/TrainConfig.kt`, `train/ScalarAdamW.kt` — 스칼라 학습 루프 + hyperparams + optimizer.
- `train/ScalarCheckpoint.kt`, `train/States.kt` — checkpoint / per-layer serialization format.
- `train/experiments/` — ConvMix*, TinyHelen*, ThreeStage*, TwoStage*, Ccmc*, Bench*, Dialogues* 진입점 32개 (turbo 백엔드).
- `sample/ScalarSampler.kt`, `sample/ChatTurbo.kt` — generation paths (scalar / turbo).
- `data/CharBPE.kt`, `data/BpePrep.kt` — tokenizer + pipeline.
- `server/InferenceApiMain.kt` — Ktor 기반 추론 HTTP API.

## Development Notes

- `src/test/kotlin/`에는 `@Test`만 둔다. 스윗 전체가 1초 이내로 끝나는 것이 원칙. `pipeline.FullPipelineTest`가 학습→체크포인트→샘플링 배관을 합성 데이터로 검증한다.
- 실제 학습/샘플링/디버그 스크립트는 `src/main/kotlin/`의 `*Main.kt` (예: `MiniTrainerMain`, `SamplerMain`, `TinyHelenSampleTurbo`)이며 `./gradlew runXxx`로 실행한다.
- 메모리 설정 가이드: `runBpe` `-Xmx12g`, `runTrainer`/turbo 학습 `-Xmx4g~8g`, 5M+ 벤치 `-Xmx6g~8g`, 그 외 대부분 `-Xmx2g`.
- 체크포인트는 가중치와 (turbo의 경우) 옵티마이저 상태를 모두 저장하므로 학습을 재개할 수 있다.
- 외부 ML 프레임워크 의존성 없음 — 텐서 연산, 그래디언트, 옵티마이저 모두 in-repo.
