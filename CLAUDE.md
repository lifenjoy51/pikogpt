# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PikoGPT is a Kotlin port of nanoGPT and micrograd by Andrej Karpathy. This is an educational project that implements a GPT model with automatic differentiation from scratch in Kotlin, without external ML libraries.

## Build and Development Commands

```bash
# Build
./gradlew build
./gradlew clean

# Run all tests (fast: 전체 1초 이내가 목표. training/sampling도 여기 포함된 smoke test 있음)
./gradlew test
./gradlew test --tests "ValueTest"
./gradlew test --tests "pipeline.FullPipelineTest"  # 합성 데이터 기반 end-to-end 배관 검증

# 실행 스크립트 — 전부 main 소스셋 + JavaExec 기반. 긴 수행 예상.
./gradlew runStoriesBpe       # BPE 토큰화 + train.bin/val.bin 생성  (-Xmx4g)
./gradlew runAlphabetPrep     # 문자 단위 데이터 준비
./gradlew runStoryGenerator   # LM Studio 연동 이야기 생성/검증     (-Xmx4g)
./gradlew runBPETest          # BPETestKt
./gradlew runMain             # MainKt
./gradlew runTrainer          # 실제 학습 (TrainerMain, -Xmx8g, 수 분~수십 분)
./gradlew runMiniTrainer      # 경량 학습 (az 알파벳)
./gradlew runSampler          # 체크포인트 로드 후 샘플링
./gradlew runAnalyzeTokens    # 토큰 분포 분석 디버그
./gradlew runDebugBPE         # BPE 디버그

# TinyHelen 전용 학습/샘플링 (1M 파라미터 벡터 백엔드)
./gradlew runTinyHelenTrainVec              # 혼합 코퍼스(book+textbook+wiki+conversation) 10k
./gradlew runTinyHelenTrainTextbookVec      # textbook-only 6k, alwaysSave=false (best avg 기반)
./gradlew runTinyHelenSampleVec             # 최신 ckpt 자동 샘플링
./gradlew runSamplePromptsFromFile \
          --args="<ckpt-dir> <prompts.txt>" # 커스텀 프롬프트 샘플링

# Resume (벡터 백엔드만)
./gradlew runTinyHelenTrainVec --args="resume"              # 최대 iter ckpt에서 이어하기
./gradlew runTinyHelenTrainVec --args="20000 resume"        # maxIters 늘려 이어하기
```

NOTE: `application` 플러그인을 쓰지 않으므로 `./gradlew run`은 없다. 위 `runXxx` 태스크를 사용한다. 역할 분리 원칙:
- **`src/test/kotlin/`** — `@Test`만. 전체 테스트 스윗이 1초 내로 끝나도록 유지.
- **`src/main/kotlin/`** — 프로덕션 코드 + 실행 스크립트(`fun main()`). 오래 걸려도 됨.

## Toolchain

- Kotlin JVM `1.9.0` (+ `kotlin("plugin.serialization") 1.9.0`)
- `org.jetbrains.kotlinx:kotlinx-serialization-json:1.5.0`
- `org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3`
- Tests: `kotlin("test")` on JUnit Platform

## Architecture

The codebase has **two parallel autodiff backends** with deliberately different priorities:

- **Scalar (`Value.kt` + `grad/` + `gpt/` + `train/` + `sample/`)** — 교육용 레퍼런스.
  모든 연산이 `Value` 스칼라 객체 그래프 위에서 일어나고, `backward()`가 위상정렬로 체인 룰을
  재생한다. micrograd의 직계 포팅. 학습이 가장 읽기 쉽다는 것이 존재 이유. 단점: 71K 파라미터
  모델이 iter당 16초. 실험용으로는 느림.
- **Vector (`vec/`)** — 실전 성능.
  `Tensor` = `FloatArray + shape`. 연산과 backward는 **autograd 없이 레이어마다 명시적**으로 기록
  (layer.forward / layer.backward를 나란히 읽으면 chain rule이 수식 그대로 드러남). 1M 파라미터 모델이
  iter당 1.7초 — 스칼라 대비 파라미터당 ~130×. MVP 수준으로만 완성 (SIMD/풀링 등은 미구현).

벡터 백엔드는 스칼라를 대체하지 않고 **병행**한다. 기존 스칼라 파일들은 수정하지 않음. 선택은
실행 시점에 `runTinyHelenTrain`(scalar) vs `runTinyHelenTrainVec`(vec)로.

이 아래 "Core components" 섹션은 두 백엔드의 파일 트리를 각각 설명한다.

### 공유 정의
- `grad/` — a micrograd-style scalar MLP (`MicrogradNeuron` → `MicrogradLayer` → `MicrogradMLP` → `MicrogradLossCalculator`). 스칼라 autodiff의 가장 단순한 예. `MlpTest` / `LossTest`로 autodiff 자체의 건강성을 검증.

### 백엔드 명명 규칙

두 백엔드가 같은 패키지 트리에 공존하므로 **모든 동명 클래스에 `Vec`/`Scalar` 접두사**를 붙여 코드만 봐도 어느 백엔드인지 분명하게 한다. 패키지 명시 없이 클래스명만으로 구분 가능. 예: `VecTrainer` vs `ScalarTrainer`, `VecSampler` vs `ScalarSampler`, `VecAdamW` vs `ScalarAdamW`, `VecPikoGPT` vs `ScalarPikoGPT`. micrograd 출처 클래스는 `Micrograd*` 접두사. `Tensor`(vec)와 `Value`(scalar)는 충돌 없어 그대로.

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
   - `ScalarCausalSelfAttention.kt` — causal multi-head attention (Q/K/V projections + masking).
   - `ScalarFeedForward.kt` — Transformer feed-forward (expand → GELU → contract).
   - `ScalarLayerNorm.kt`, `ScalarDropout.kt`, `ScalarLinear.kt`, `ScalarEmbeddingTable.kt`, `Sequence.kt`, `Logits.kt`
   - `Matrix.kt` — type-safe matrix abstractions; the shape-safety foundation used across the GPT stack.

4. **Training (`src/main/kotlin/train/`)**
   - `ScalarTrainer.kt` — 스칼라 백엔드 학습 루프: LR schedule, grad clipping, eval, checkpointing.
   - `ScalarAdamW.kt` — 스칼라 백엔드 옵티마이저 (`Value` 단위).
   - `DataLoader.kt` — minibatch sampling over `train.bin` / `val.bin`. 두 백엔드 공유.
   - `TrainConfig.kt` — hyperparameters + data/model paths.
   - `ScalarCheckpoint.kt` — serializable wrapper (iteration, best loss, model args, optimizer state).
   - `States.kt` — per-layer serializable state DTOs used by `ScalarCheckpoint`.
   - `experiments/` — 14개 실험 진입점 (`ConvMix*TrainVec.kt`, `TinyHelenTrain*.kt` — SwiGLU/RoPE 변형 포함). 코어 학습 로직과 분리.

5. **Sampling (`src/main/kotlin/sample/`)**
   - `ScalarSampler.kt` — 스칼라 백엔드. 체크포인트 로드 + 텍스트 생성 (temperature / top-k).
   - `SampleConfig.kt` — sampling parameters (long-name canonical: `modelDirectoryPath`, `numberOfSamples`, ...).

6. **Data processing (`src/main/kotlin/data/`)**
   - `SimpleBPE.kt` — BPE train + encode/decode. 플래그: `lowercase`, `useWordPreTokenize` (GPT-2 스타일 regex 사전 분할), `standardBpeScoring`(빈도 기준 merge), `verbose`. 학습된 상태를 `getMerges()`로 내보내고 `restore(stoi, merges)`로 복원 — Sampler가 학습과 **정확히 같은 토큰화를 재생**.
   - `StoriesBpePrep.kt` — BPE 전처리 파이프라인. `main(args)` CLI로 데이터 경로 지정. meta.json에 `merges` + 플래그까지 저장.
   - `AlphabetPrep.kt` — alternative character-level preprocessing path. Has a `main()`.
   - `StoryGenerator.kt` — optional LM Studio integration (`http://127.0.0.1:1234`, `google/gemma-3-1b`) for generating/validating children's stories. Has a `main()`.
   - `MetaInfo.kt` — vocab metadata DTO. `merges`, `lowercase`, `useWordPreTokenize`, `specialTokens` 포함 (모두 기본값 있어 구 meta.json도 그대로 로드).

7. **Vector backend (`src/main/kotlin/vec/`)**
   - `Tensor.kt` — `shape: IntArray`, `data: FloatArray`, lazy `grad: FloatArray?`. 계산 그래프 없음.
   - `ops/` — 원자 연산. 각 파일에 forward + backward + 수식 주석이 함께.
     `MatMul.kt`, `Softmax.kt`, `GELU.kt`, `SiLU.kt`(SwiGLU용), `LayerNormOp.kt`, `CrossEntropy.kt`, `RoPE.kt`(rotary position embedding).
   - `layer/` — 7개 레이어가 각자 `forward(x): Tensor`와 `backward(gy): Tensor`를 노출. 모두 `Vec` 접두사:
     `VecLinear`, `VecLayerNorm`, `VecMLP`(GELU/SwiGLU 분기), `VecSelfAttention`(multi-head causal + RoPE 옵션), `VecTransformerBlock`(pre-LN), `VecEmbeddingTable`, `VecPikoGPT`(RoPE 모드면 position embedding 제외).
   - `VecAdamW.kt`, `VecTrainer.kt`, `VecSampler.kt`, `VecCheckpoint.kt` — 파라미터별 FloatArray 기반의 옵티마이저 + 학습 루프 + 샘플러. 체크포인트는 `model/<dataset>/vec/<paramCount>/v0001/`(zero-pad 4자리 버전 번호)에 저장. 레거시 `bestLoss*10` 정수 디렉터리도 loader가 함께 인식.
   - `Parallel.kt` — 데이터 병렬 학습 helper.
   - 테스트: `src/test/kotlin/vec/` — 각 op의 수치(유한차분) gradient 검사 + 레이어 단위 dx 검증 + `VFullPipelineTest`(합성 vocab end-to-end) + `AdamWTest`.

### Design patterns

- **Pure Kotlin, no ML libs.** Everything (autodiff, layers, optimizer, tokenizer) is implemented in-repo.
- **Scalar autodiff graph.** `Value` builds a dynamic computation graph per forward pass; `backward()` walks it in reverse-topological order.
- **Serialization via kotlinx.serialization.** Configs, checkpoints, and per-layer `States.kt` DTOs are `@Serializable`; model weights go to a separate `.bin`.
- **Execution convention.** 모든 실행 스크립트는 `src/main/kotlin/`의 `fun main()`이며 `build.gradle.kts`의 `runXxx` `JavaExec` 태스크로 실행한다 (`train/TrainerMain.kt`, `train/MiniTrainerMain.kt`, `sample/SamplerMain.kt`, `data/AnalyzeTokensMain.kt` 등). 통합 CLI는 없다. 테스트 디렉토리에는 `@Test`만 둔다.

## Training Workflow

1. **Data preparation** — `StoriesBpePrep.kt` (or `AlphabetPrep.kt`) tokenizes text into `data/[dataset]/train.bin` + `val.bin` + `meta.json`. `./gradlew runStoriesBpe` / `runAlphabetPrep`.
2. **Training** — `./gradlew runTrainer` (스모크 50 iter 프리셋, `train.TrainerMain`)이나 `runMiniTrainer`. 체크포인트는 최적 검증 손실이 갱신될 때마다 저장된다.
3. **Checkpoint layout** — 벡터 백엔드(`VecTrainer`)는 `${config.modelDir}/${datasetName}/vec/${paramCount}/v0001/` (4자리 zero-pad 버전 번호) 경로에 `checkpoint.json`, `model_weights.bin`, `meta.json`, `optimizer_state.bin`을 쓴다. 매 저장마다 +1. 레거시 `bestLoss*10` 정수 디렉터리는 보존되며 loader가 인식.
4. **Sampling** — `./gradlew runSampler`로 체크포인트 디렉토리를 로드해 텍스트 생성 (`sample.SamplerMain`).

## Data Layout

```
data/
├── [dataset]/                # e.g. 1k, simple, tinyhelen, tinyhelen-textbook
│   ├── stories.txt           # 단일 입력 (fallback: 90:10 cut)
│   ├── train.txt / val.txt   # 분리 입력 (있으면 우선, vocab은 train에서만 학습 — leakage 차단)
│   ├── meta.json             # vocab size + token<->id maps + BPE merges
│   ├── train.bin             # tokenized training set
│   ├── val.bin               # tokenized validation set
│   └── unique_words.txt      # optional vocab dump

model/                        # 모든 체크포인트 루트 (gitignored)
└── [datasetName]/            # config.dataPath 마지막 segment (예: tinyhelen, tinyhelen-textbook)
    └── vec/                  # 백엔드 구분 (스칼라는 직접 파라미터 수부터; 곧 통일 예정)
        └── [paramCount]/     # 예: 1057536
            └── v0001/        # 4자리 zero-pad 버전 (매 저장마다 +1). 레거시 "28" 같은 bestLoss*10도 인식.
                ├── checkpoint.json   # iteration, best loss, model args
                ├── meta.json         # copied from the data dir
                ├── model_weights.bin # serialized weights (vec: big-endian float32)
                └── optimizer_state.bin # vec 전용: AdamW timeStep + 모멘트 (resume용)
```

## Key Files to Understand

- `Value.kt` — autodiff foundation.
- `gpt/Matrix.kt` — shape-safe matrix layer underpinning the GPT stack.
- `gpt/ScalarPikoGPT.kt`, `gpt/GPTConfig.kt` — 스칼라 백엔드 모델 아키텍처 + config.
- `gpt/ScalarTransformerBlock.kt`, `gpt/ScalarCausalSelfAttention.kt`, `gpt/ScalarFeedForward.kt` — block internals (scalar).
- `vec/layer/VecPikoGPT.kt`, `vec/VecTrainer.kt` — 벡터 백엔드 모델 + 학습 루프 (실전 성능, SwiGLU/RoPE 지원).
- `train/ScalarTrainer.kt`, `train/TrainConfig.kt`, `train/ScalarAdamW.kt` — 스칼라 학습 루프 + hyperparams + optimizer.
- `train/ScalarCheckpoint.kt`, `train/States.kt` — checkpoint / per-layer serialization format.
- `train/experiments/` — ConvMix*, TinyHelen* 실험 진입점 14개 (별도 서브패키지).
- `sample/ScalarSampler.kt`, `sample/ChatVec.kt` — generation paths (scalar / vec).
- `data/SimpleBPE.kt`, `data/StoriesBpePrep.kt` — tokenizer + pipeline.

## External Dependency

`StoryGenerator` requires a locally running LM Studio at `http://127.0.0.1:1234` serving `google/gemma-3-1b`. Other flows do **not** need it.

## Development Notes

- `src/test/kotlin/`에는 `@Test`만 둔다. 스윗 전체가 1초 이내로 끝나는 것이 원칙. `pipeline.FullPipelineTest`가 학습→체크포인트→샘플링 배관을 합성 데이터로 검증한다.
- 실제 학습/샘플링/디버그 스크립트는 `src/main/kotlin/`의 `*Main.kt` (예: `TrainerMain`, `SamplerMain`, `MiniTrainerMain`)이며 `./gradlew runXxx`로 실행한다.
- `runStoriesBpe`/`runStoryGenerator`는 `-Xmx4g`, `runTrainer`는 `-Xmx8g`, 나머지는 `-Xmx2g`.
- 체크포인트는 가중치와 옵티마이저 상태를 모두 저장하므로 학습을 재개할 수 있다.
- 외부 ML 프레임워크 의존성 없음 — 텐서 연산, 그래디언트, 옵티마이저 모두 in-repo.
